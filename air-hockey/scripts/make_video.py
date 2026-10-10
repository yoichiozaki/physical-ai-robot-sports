"""Compose the submission video  videos/PAI最終課題_yoichiozaki.mp4  (about 150 s, 1280x720, H.264).

Every match clip is a deterministic replay of a real simulation (seed + controllers); all
overlays are the controllers' actual outputs and every number shown on the result cards is
read from results/eval_main. The script also writes videos/video_manifest.json (segment list,
seeds, sources) and the per-clip simulation logs (videos/clips/*.json).

    python scripts/make_video.py
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFilter

from hockey.controllers.factory import make_controller
from hockey.controllers.opponents import make_opponent
from hockey.evaluation.conditions import CONDITIONS
from hockey.evaluation.match import StepRecord, play_match
from hockey.observation import Geometry
from hockey.simulation import HockeySim
from hockey.simulation.game import GameConfig
from hockey.visualization.hud import font
from hockey.visualization.recorder import ClipRecorder, RecorderConfig
from hockey.visualization.scenes import scene_setup
from hockey.visualization.video import VideoWriter, probe

W, H, FPS = 1280, 720, 30
OUT = Path("videos/PAI最終課題_yoichiozaki.mp4")
CLIPS = Path("videos/clips")
FIG = Path("results/figures")
EVAL = Path(os.environ.get("EVAL_DIR", "results/eval_main"))
_HYB = os.environ.get("HYBRID_MODEL", "checkpoints/ppo_hybrid_s0/final_model.zip")
HYBRID = f"hybrid:{_HYB}"
HYBRID_ADAPT = f"hybrid_adaptive:{_HYB}"
PPO = "ppo:" + os.environ.get("PPO_MODEL", "checkpoints/ppo_direct_s0/final_model.zip")
NAMES = {
    "reactive": "A: 追従制御",
    "physics": "B: 物理予測制御",
    "physics_adaptive": "B*: 物理+適応",
    HYBRID: "D: Hybrid AI",
    HYBRID_ADAPT: "D*: Hybrid AI+適応",
    PPO: "C: 純粋RL",
}
OPP_JA = {
    "stationary": "静止",
    "random": "ランダム",
    "reactive": "追従型",
    "defensive": "守備型",
    "aggressive": "攻撃型",
}

manifest: list[dict] = []


# ----------------------------------------------------------------------------- cards
def _canvas(bg: np.ndarray | None = None) -> Image.Image:
    if bg is not None:
        img = Image.fromarray(bg).filter(ImageFilter.GaussianBlur(6))
        dark = Image.new("RGB", img.size, (8, 12, 22))
        return Image.blend(img, dark, 0.72)
    img = Image.new("RGB", (W, H), (14, 20, 34))
    d = ImageDraw.Draw(img)
    for i in range(H):  # subtle vertical gradient
        c = int(14 + 18 * i / H)
        d.line([(0, i), (W, i)], fill=(c, c + 6, c + 18))
    return img


def card(
    title: str,
    lines: list[str],
    seconds: float,
    name: str,
    bg: np.ndarray | None = None,
    subtitle: str = "",
    image: Path | None = None,
    image_box: tuple[int, int, int, int] = (90, 150, 1190, 660),
    footer: str = "",
) -> Path:
    img = _canvas(bg)
    d = ImageDraw.Draw(img)
    d.text((70, 52), title, font=font(40), fill=(245, 247, 250))
    if subtitle:
        d.text((72, 108), subtitle, font=font(22), fill=(150, 200, 255))
    if image is not None and image.exists():
        fig = Image.open(image).convert("RGB")
        x0, y0, x1, y1 = image_box
        fig.thumbnail((x1 - x0, y1 - y0), Image.LANCZOS)
        panel = Image.new("RGB", (fig.width + 16, fig.height + 16), (250, 250, 252))
        panel.paste(fig, (8, 8))
        img.paste(panel, (x0 + (x1 - x0 - panel.width) // 2, y0 + (y1 - y0 - panel.height) // 2))
    y = 170 if subtitle else 140
    for line in lines:
        size = 26 if not line.startswith("  ") else 22
        color = (235, 238, 242) if not line.startswith("  ") else (180, 190, 205)
        d.text((90, y), line, font=font(size), fill=color)
        y += 46 if size == 26 else 38
    if footer:
        d.text((70, H - 50), footer, font=font(18), fill=(150, 160, 175))
    path = CLIPS / f"{name}.mp4"
    frame = np.asarray(img)
    n = int(seconds * FPS)
    with VideoWriter(path, W, H, FPS) as vw:
        for i in range(n):
            a = min(1.0, i / 8, (n - 1 - i) / 8)
            vw.write((frame * max(a, 0.0)).astype(np.uint8))
    manifest.append({"segment": name, "type": "card", "seconds": seconds, "title": title})
    return path


def table_card(
    title: str,
    header: list[str],
    rows: list[list[str]],
    seconds: float,
    name: str,
    footer: str = "",
    highlight: set[int] | None = None,
) -> Path:
    img = _canvas()
    d = ImageDraw.Draw(img)
    d.text((70, 52), title, font=font(36), fill=(245, 247, 250))
    col_x = [90, 470, 640, 800, 960]
    y = 150
    for x, h in zip(col_x, header, strict=True):
        d.text((x, y), h, font=font(22), fill=(150, 200, 255))
    y += 48
    for i, row in enumerate(rows):
        if highlight and i in highlight:
            d.rounded_rectangle((75, y - 8, 1200, y + 36), radius=8, fill=(30, 60, 40))
        for x, cell in zip(col_x, row, strict=True):
            d.text((x, y), cell, font=font(24), fill=(235, 238, 242))
        y += 52
    if footer:
        d.text((70, H - 50), footer, font=font(18), fill=(150, 160, 175))
    path = CLIPS / f"{name}.mp4"
    frame = np.asarray(img)
    n = int(seconds * FPS)
    with VideoWriter(path, W, H, FPS) as vw:
        for i in range(n):
            a = min(1.0, i / 8, (n - 1 - i) / 8)
            vw.write((frame * max(a, 0.0)).astype(np.uint8))
    manifest.append(
        {"segment": name, "type": "table", "seconds": seconds, "title": title, "rows": rows}
    )
    return path


# ----------------------------------------------------------------------------- clips
@dataclass
class Clip:
    name: str
    home: str
    away: str
    seed: int
    t0: float
    t1: float
    condition: str = "nominal"
    speed: float = 1.0
    camera: str = "broadcast"
    title: str = ""
    captions: tuple[tuple[float, str], ...] = ()
    away_overlays: bool = False
    first: str | None = None
    lines_fn: Callable[[object], list[str]] | None = None
    scene: str = "match"


def find_events(home: str, away: str, seed: int, condition: str, duration: float) -> list[dict]:
    """Replay a match without rendering and list goals with the scorer's strategy."""
    sim = HockeySim()
    geom = Geometry.from_sim_config(sim.cfg)
    events: list[dict] = []
    last_strategy = {"v": ""}

    def on_step(rec: StepRecord) -> None:
        if rec.home_cmd.info.mode == "strike":
            last_strategy["v"] = rec.home_cmd.info.strategy
        if rec.event in ("goal_home", "goal_away"):
            events.append({"t": rec.sim.time, "event": rec.event, "strategy": last_strategy["v"]})

    play_match(
        sim, make_controller(home, geom), make_opponent(away, geom), seed, CONDITIONS[condition],
        GameConfig(match_duration=duration), on_step=on_step, first="home",
    )  # fmt: skip
    return events


def render_clip(c: Clip) -> Path:
    sim = HockeySim()
    geom = Geometry.from_sim_config(sim.cfg)
    home = make_controller(c.home, geom)
    away = make_opponent(c.away, geom)
    cfg = RecorderConfig(
        width=W, height=H, fps=FPS, speed=c.speed, camera=c.camera,
        home_name=NAMES.get(c.home, c.home), away_name=f"相手: {OPP_JA.get(c.away, c.away)}",
        overlays_away=c.away_overlays, title=c.title,
    )  # fmt: skip

    def caption(t: float) -> str:
        text = ""
        for start, s in c.captions:
            if t + c.t0 >= start:
                text = s
        return text

    path = CLIPS / f"{c.name}.mp4"
    with VideoWriter(path, W, H, FPS) as vw:
        rec = ClipRecorder(vw, cfg, caption_fn=caption)

        def on_step(r: StepRecord) -> None:
            if r.sim.time < c.t0:
                return
            if c.lines_fn is not None:
                rec.cfg.extra_lines = c.lines_fn(home)
            rec(r)

        try:
            res = play_match(
                sim, home, away, c.seed, CONDITIONS[c.condition], GameConfig(match_duration=c.t1),
                on_step=on_step, first=c.first, setup=scene_setup(c.scene),
            )  # fmt: skip
        finally:
            rec.close()
    meta = {k: v for k, v in c.__dict__.items() if k != "lines_fn"}
    rec.save_log(path.with_suffix(".json"), {"clip": meta, "summary": res.summary()})
    manifest.append(
        {
            "segment": c.name, "type": "simulation", "home": c.home, "away": c.away, "seed": c.seed,
            "condition": c.condition, "game_time": [c.t0, c.t1], "playback_speed": c.speed,
            "seconds": probe(path)["duration"],
        }
    )  # fmt: skip
    return path


def first_frame(path: Path) -> np.ndarray:
    tmp = path.with_suffix(".png")
    subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-ss", "0.5", "-i", str(path), "-frames:v", "1", str(tmp)],
                   check=True)  # fmt: skip
    return np.asarray(Image.open(tmp).convert("RGB"))


def concat(paths: list[Path], out: Path) -> None:
    lst = CLIPS / "concat.txt"
    lst.write_text("".join(f"file '{p.resolve().as_posix()}'\n" for p in paths), encoding="utf-8")
    out.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["ffmpeg", "-loglevel", "error", "-y", "-f", "concat", "-safe", "0", "-i", str(lst), "-c", "copy",
         "-movflags", "+faststart", str(out)],
        check=True,
    )  # fmt: skip


# ----------------------------------------------------------------------------- results text
def result_rows() -> tuple[list[list[str]], set[int]]:
    df = pd.read_csv(EVAL / "matches.csv")
    nom = df[df["condition"] == "nominal"]
    rows, wins = [], []
    order = ["A", "B", "B+", "B*", "C", "D", "D*"]
    for name, g in sorted(
        nom.groupby("controller"), key=lambda kv: order.index(kv[0].split(":")[0])
    ):
        thr = g["threats"].sum()
        rows.append(
            [
                name,
                f"{g['win'].mean() * 100:.0f}%",
                f"{g['goals_for'].mean():.2f}",
                f"{g['goals_against'].mean():.2f}",
                f"{g['saves'].sum() / thr * 100:.0f}%" if thr else "—",
            ]
        )
        wins.append(g["win"].mean())
    best = {int(np.argmax(wins))} if wins else set()
    return rows, best


# ----------------------------------------------------------------------------- main
def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--out", default=str(OUT))
    a = p.parse_args(argv)
    CLIPS.mkdir(parents=True, exist_ok=True)
    segs: list[Path] = []

    # 1) Opening highlight: a Hybrid-AI goal (prefer a bank shot) against the aggressive opponent
    best = None
    for seed in range(20000, 20012):
        for ev in find_events(HYBRID, "aggressive", seed, "nominal", 40.0):
            if ev["event"] == "goal_home" and ev["t"] > 6.0:
                score = 2 if "Bank" in ev["strategy"] else 1
                if best is None or score > best[0]:
                    best = (score, seed, ev["t"], ev["strategy"])
        if best and best[0] == 2:
            break
    seed, tg = (best[1], best[2]) if best else (20000, 12.0)
    hl = Clip("01_highlight", HYBRID, "aggressive", seed, max(0.0, tg - 7.0), tg + 1.5, first="home",
              title="Adaptive Air Hockey AI",
              captions=((0.0, "物理予測 × 強化学習で戦い方を変えるエアホッケーロボット"),
                        (tg - 2.5, "AI が狙いと打ち方を選び、ロボットが物理制約の中で実行する")))  # fmt: skip
    hl_path = render_clip(hl)
    hl_log = json.loads(hl_path.with_suffix(".json").read_text(encoding="utf-8"))
    if not any(st["referee"] == "goal_home" for st in hl_log["steps"]):
        raise SystemExit("highlight replay did not reproduce the goal (non-deterministic setup?)")
    segs.append(hl_path)
    bg = first_frame(hl_path)

    # 2) Problem statement
    segs.append(card(
        "課題と研究課題", [
            "・パックは 2–5 m/s で壁に反射しながら飛ぶ → 0.3–0.6 秒で判断・移動が必要",
            "・ロボットは関節とモーターで動く物理機械（速度・加速度・可動範囲に限界）",
            "・相手の戦い方はさまざま（静止・ランダム・追従・守備型・攻撃型）",
            "",
            "研究課題: 物理モデルによる予測制御に比べ、機械学習による",
            "戦略選択・予測補正は対戦性能と未知環境への適応を改善できるか？",
        ], 16, "02_problem", bg=bg, subtitle="Physical AI 応用1 最終課題 / OmniCampus: yoichiozaki",
    ))  # fmt: skip

    # 3) Architecture + robot mechanism
    segs.append(card("システム構成", [], 11, "03_architecture", image=FIG / "architecture.png",
                     image_box=(40, 120, 1240, 690)))  # fmt: skip
    segs.append(render_clip(Clip(
        "04_robot", "physics", "stationary", 3, 0.0, 2.6, speed=0.5, camera="home_view", scene="strike",
        title="ロボット機構（スロー 0.5x）",
        captions=((0.0, "2 軸ガントリー: X(ブリッジ)・Y(キャリッジ) の直動関節 + 力制限付き速度サーボ"),
                  (1.2, "打撃は「助走位置 → 時刻指定で加速 → 打点を通過」の直線軌道で実行")),
    )))  # fmt: skip

    # 4) Controller comparison on the same opponent and seed
    cmp_seed = 30001
    for i, (spec, cap) in enumerate(
        (
            ("reactive", "A: 従来の追従制御 — 現在位置を追うだけでは速いシュートに間に合わない"),
            ("physics", "B: 物理予測制御 — 反射を含む軌道を予測し、到達可能な点で迎撃・打撃"),
            (HYBRID, "D: Hybrid AI — 物理予測の上で、強化学習が戦略(狙い・壁打ち・速度)を選ぶ"),
        )
    ):
        segs.append(render_clip(Clip(f"05_compare_{i}", spec, "aggressive", cmp_seed, 0.0, 10.0, first="away",
                                     title="同じ相手・同じ初期条件で比較", captions=((0.0, cap),))))  # fmt: skip

    # 5) Advanced: wall-bounce prediction (slow, top view), strategy adaptation, unknown table
    segs.append(render_clip(Clip(
        "06_bounce", "physics", "stationary", 30002, 0.0, 1.6, speed=0.4, camera="top", scene="bank_defense",
        title="壁反射の予測（スロー 0.4x、真上から）",
        captions=((0.0, "緑線: 減衰と壁反射を含む予測軌道 / 橙: 予測した防御点・打撃点（実際のアルゴリズム出力）"),),
    )))  # fmt: skip
    for i, opp in enumerate(("stationary", "defensive")):
        segs.append(render_clip(Clip(
            f"07_adapt_{opp}", HYBRID, opp, 30010 + i, 0.0, 9.0, first="home",
            title=f"相手に応じた戦略選択（相手: {OPP_JA[opp]}）",
            captions=((0.0, "AI は相手マレットの位置・動きを見て、空いているコースや壁打ちを選ぶ"),),
        )))  # fmt: skip

    def est_lines(ctrl) -> list[str]:
        ex = getattr(getattr(ctrl, "adapter", None), "executor", None)
        est = getattr(ex, "estimator", None)
        if est is None:
            return []
        m = est.model
        return [f"推定 減衰率|{m.decay:.2f} /s", f"推定 反発係数|{m.wall_restitution:.2f}"]

    segs.append(render_clip(Clip(
        "08_unknown", HYBRID_ADAPT, "reactive", 30020, 0.0, 10.0, condition="unknown_sticky", first="away",
        title="未知条件（高摩擦・低反発 + ノイズ）",
        captions=((0.0, "学習時に見ていない物理条件。オンライン同定で減衰率・反発係数を推定し予測を補正"),),
        lines_fn=est_lines,
    )))  # fmt: skip

    # 6) Results (numbers from results/eval_main)
    rows, best = result_rows()
    segs.append(table_card("定量評価（標準条件, 相手 5 種 × 20 試合 × 60 秒）", ["制御器", "勝率", "得点/60s", "失点/60s", "防御成功"],
                           rows, 10, "09_results_table", highlight=best,
                           footer="出典: results/eval_main/matches.csv（再生成: python scripts/evaluate.py）"))  # fmt: skip
    for name, fig, sec in (("10_win_rate", "win_rate.png", 7), ("11_generalization", "generalization.png", 8),
                           ("12_strategy", "strategy_usage.png", 6)):  # fmt: skip
        segs.append(card({"10_win_rate": "勝率の比較", "11_generalization": "未知条件への汎化",
                          "12_strategy": "相手ごとの戦略選択"}[name], [], sec, name, image=FIG / fig))  # fmt: skip
    conclusion = json.loads(Path("results/conclusion.json").read_text(encoding="utf-8"))[
        "video_lines"
    ]
    segs.append(card("結論と今後の課題", conclusion, 14, "13_conclusion", bg=bg,
                     footer="コード: GitHub yoichiozaki/physical-ai-robot-sports（air-hockey）／ 全映像はシミュレーションの実録（演出用の予測表示なし）"))  # fmt: skip

    concat(segs, Path(a.out))
    info = probe(a.out)
    (Path("videos") / "video_manifest.json").write_text(
        json.dumps(
            {"output": str(a.out), "probe": info, "segments": manifest},
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    print(json.dumps(info, ensure_ascii=False))
    if shutil.which("ffprobe") and not 60 <= info["duration"] <= 180:
        raise SystemExit(f"video duration {info['duration']:.1f}s outside 60-180 s")


if __name__ == "__main__":
    main()
