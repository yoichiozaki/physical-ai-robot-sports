"""Compose the demo video  videos/PAI最終課題_yoichiozaki.mp4  (about 150 s, 1280x720, H.264).

Every match clip is a deterministic replay of a real simulation (seed + controllers); all
overlays are the controllers' actual outputs and every number shown on the result cards is
read from results/eval_main. The script also writes videos/video_manifest.json (segment list,
seeds, sources) and per-clip simulation logs (videos/clips/*.json).

    python scripts/make_video.py
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFilter

from tennis.control.factory import make_controller
from tennis.control.players import opponent_factory
from tennis.evaluation.conditions import CONDITIONS
from tennis.evaluation.match import Match, StepRecord
from tennis.sim.game import GameConfig
from tennis.sim.physics import TennisSim
from tennis.viz.hud import font
from tennis.viz.recorder import ClipRecorder, RecorderConfig, adaptation_lines
from tennis.viz.video import VideoWriter, probe

W, H, FPS = 1280, 720, 30
OUT = Path("videos/PAI最終課題_yoichiozaki.mp4")
CLIPS = Path("videos/clips")
FIG = Path("results/figures")
EVAL = Path(os.environ.get("EVAL_DIR", "results/eval_main"))
HYBRID = "hybrid:" + os.environ.get("HYBRID_MODEL", "checkpoints/ppo_hybrid/best_model.zip")
LEAGUE_D = "hybrid:checkpoints/ppo_league/league/gen_000.zip"  # = D (league generation 0)
SB_D0 = "hybrid:checkpoints/ppo_league_sb/league/gen_000.zip"
SB_CHAMP = "hybrid:checkpoints/ppo_league_sb/league/gen_006.zip"
NAMES = {
    "naive": "A: 素朴な制御",
    "physics": "B: 物理予測制御",
    "physics_rule": "B+: 物理+ルール戦術",
    "physics_adaptive": "B*: 物理+オンライン適応",
    HYBRID: "D: Hybrid AI",
    LEAGUE_D: "D: Hybrid AI",
    SB_CHAMP: "自己対戦 最終世代",
}
OPP_JA = {
    "steady": "安定型",
    "random": "ランダム",
    "aggressive": "攻撃型",
    "lob": "ロブ",
    "slow": "低速",
}
COND_JA = {
    "nominal": "標準",
    "randomized": "ランダム化",
    "unknown_wind": "未知: 強風",
    "unknown_court": "未知: 速いコート",
}

manifest: list[dict] = []


# ----------------------------------------------------------------------------- cards
def _canvas(bg: np.ndarray | None = None) -> Image.Image:
    if bg is not None:
        img = Image.fromarray(bg).filter(ImageFilter.GaussianBlur(6))
        return Image.blend(img, Image.new("RGB", img.size, (8, 12, 22)), 0.74)
    img = Image.new("RGB", (W, H), (14, 20, 34))
    d = ImageDraw.Draw(img)
    for i in range(H):
        c = int(14 + 18 * i / H)
        d.line([(0, i), (W, i)], fill=(c, c + 6, c + 18))
    return img


def _write_still(img: Image.Image, seconds: float, name: str) -> Path:
    path = CLIPS / f"{name}.mp4"
    frame = np.asarray(img)
    n = int(seconds * FPS)
    with VideoWriter(path, W, H, FPS) as vw:
        for i in range(n):
            a = min(1.0, i / 8, (n - 1 - i) / 8)
            vw.write((frame * max(a, 0.0)).astype(np.uint8))
    return path


def card(title: str, lines: list[str], seconds: float, name: str, bg: np.ndarray | None = None,
         subtitle: str = "", image: Path | None = None,
         image_box: tuple[int, int, int, int] = (90, 150, 1190, 660), footer: str = "") -> Path:  # fmt: skip
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
        small = line.startswith("  ")
        d.text((90, y), line, font=font(22 if small else 26),
               fill=(180, 190, 205) if small else (235, 238, 242))  # fmt: skip
        y += 38 if small else 46
    if footer:
        d.text((70, H - 50), footer, font=font(18), fill=(150, 160, 175))
    manifest.append({"segment": name, "type": "card", "seconds": seconds, "title": title})
    return _write_still(img, seconds, name)


def table_card(title: str, header: list[str], rows: list[list[str]], seconds: float, name: str,
               col_x: list[int], footer: str = "", highlight: set[int] | None = None) -> Path:  # fmt: skip
    img = _canvas()
    d = ImageDraw.Draw(img)
    d.text((70, 52), title, font=font(34), fill=(245, 247, 250))
    y = 150
    for x, h in zip(col_x, header, strict=True):
        d.text((x, y), h, font=font(21), fill=(150, 200, 255))
    y += 50
    for i, row in enumerate(rows):
        if highlight and i in highlight:
            d.rounded_rectangle((75, y - 8, 1210, y + 36), radius=8, fill=(30, 60, 40))
        for x, cell in zip(col_x, row, strict=True):
            d.text((x, y), cell, font=font(23), fill=(235, 238, 242))
        y += 52
    if footer:
        d.text((70, H - 50), footer, font=font(18), fill=(150, 160, 175))
    manifest.append(
        {"segment": name, "type": "table", "seconds": seconds, "title": title, "rows": rows}
    )
    return _write_still(img, seconds, name)


# ----------------------------------------------------------------------------- clips
@dataclass
class Clip:
    name: str
    home: str
    away: str
    seed: int
    t0: float  # game-time window [t0, t1]
    t1: float
    points: int  # points to simulate (the window must lie inside them)
    condition: str = "nominal"
    speed: float = 1.0
    camera: str = "behind_home"
    title: str = ""
    captions: tuple[tuple[float, str], ...] = ()  # (game time, caption)
    first: str = "home"
    adapt_lines: bool = False
    service_line: float | None = None  # tennis service-box rule (GameConfig.service_line)
    away_name: str | None = None  # label of a learned opponent


def _players(home_spec: str, away: str, sim: TennisSim):
    """``away`` is a scripted opponent name or a controller spec (e.g. hybrid:<policy.zip>)."""
    if ":" in away:
        return make_controller(home_spec, sim.cfg), make_controller(away, sim.cfg, "away")
    return make_controller(home_spec, sim.cfg), opponent_factory(sim.cfg)[away]()


def find_points(home_spec: str, away: str, seed: int, condition: str, points: int,
                first: str = "home", service_line: float | None = None) -> list[dict]:  # fmt: skip
    """Replay a match without rendering: per point start/end time, winner, rally length and
    the times of the home robot's racket contacts."""
    sim = TennisSim()
    home, opp = _players(home_spec, away, sim)
    out: list[dict] = []
    cur = {"t_start": 0.0, "home_hits": []}

    def on_step(rec: StepRecord) -> None:
        for e in rec.result.events:
            if e.kind == "racket_home":
                cur["home_hits"].append(e.time)
        if rec.point is not None:
            p = rec.point
            out.append({**cur, "t_end": rec.sim.time, "winner": p.winner, "reason": p.reason,
                        "hits": p.hits, "index": len(out)})  # fmt: skip
            cur.update({"t_start": rec.sim.time, "home_hits": []})

    Match(sim, home, opp, seed, CONDITIONS[condition],
          GameConfig(points=points, service_line=service_line), on_step, first_server=first).run()  # fmt: skip
    return out


def render_clip(c: Clip) -> Path:
    path = CLIPS / f"{c.name}.mp4"
    meta = dict(c.__dict__.items())
    log_path = path.with_suffix(".json")
    learned = c.home.startswith(("hybrid", "direct"))  # checkpoint may change under the same name
    if os.environ.get("REUSE_CLIPS") and not learned and path.exists() and log_path.exists():
        old = json.loads(log_path.read_text(encoding="utf-8"))["meta"]["clip"]
        if json.loads(json.dumps(meta)) == old:
            manifest.append(
                {
                    "segment": c.name, "type": "simulation", "home": c.home, "away": c.away,
                    "seed": c.seed, "condition": c.condition,
                    "game_time": [round(c.t0, 3), round(c.t1, 3)], "playback_speed": c.speed,
                    "seconds": probe(path)["duration"], "reused": True,
                }
            )  # fmt: skip
            return path
    sim = TennisSim()
    home, opp = _players(c.home, c.away, sim)
    cfg = RecorderConfig(
        width=W, height=H, fps=FPS, speed=c.speed, camera=c.camera,
        home_name=NAMES.get(c.home, c.home),
        away_name=c.away_name or f"相手: {OPP_JA.get(c.away, c.away)}",
        title=c.title,
    )  # fmt: skip

    def caption(t: float) -> str:
        text = ""
        for start, s in c.captions:
            if t + c.t0 >= start:
                text = s
        return text

    with VideoWriter(path, W, H, FPS) as vw:
        rec = ClipRecorder(vw, cfg, caption_fn=caption,
                           lines_fn=(lambda: adaptation_lines(home)) if c.adapt_lines else None)  # fmt: skip

        def on_step(r: StepRecord) -> None:
            if c.t0 <= r.sim.time <= c.t1:
                rec(r)

        game = GameConfig(points=c.points, service_line=c.service_line)
        m = Match(sim, home, opp, c.seed, CONDITIONS[c.condition], game, on_step,
                  first_server=c.first)  # fmt: skip
        try:
            while not m.done and sim.time <= c.t1:
                m.observe()
                m.advance()
        finally:
            rec.close()
    rec.save_log(log_path, {"clip": meta})
    manifest.append(
        {
            "segment": c.name, "type": "simulation", "home": c.home, "away": c.away, "seed": c.seed,
            "condition": c.condition, "game_time": [round(c.t0, 3), round(c.t1, 3)],
            "playback_speed": c.speed, "seconds": probe(path)["duration"],
        }
    )  # fmt: skip
    return path


def first_frame(path: Path) -> np.ndarray:
    tmp = path.with_suffix(".png")
    subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-ss", "0.5", "-i", str(path),
                    "-frames:v", "1", str(tmp)], check=True)  # fmt: skip
    return np.asarray(Image.open(tmp).convert("RGB"))


def concat(paths: list[Path], out: Path) -> None:
    lst = CLIPS / "concat.txt"
    lst.write_text("".join(f"file '{p.resolve().as_posix()}'\n" for p in paths), encoding="utf-8")
    out.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-f", "concat", "-safe", "0", "-i", str(lst),
                    "-c", "copy", "-movflags", "+faststart", str(out)], check=True)  # fmt: skip


# ----------------------------------------------------------------------------- selection
def best_rally(home: str, opponents: tuple[str, ...], seeds: range, condition: str, points: int,
               min_hits: int = 7) -> tuple[str, int, dict]:  # fmt: skip
    """The longest rally won by the home robot (deterministic replays, no rendering)."""
    best = None
    for seed in seeds:
        for opp in opponents:
            for p in find_points(home, opp, seed, condition, points):
                ok = p["winner"] == "home" and p["t_end"] - p["t_start"] < 16
                if ok and (best is None or p["hits"] > best[2]["hits"]):
                    best = (opp, seed, p)
            if best is not None and best[2]["hits"] >= min_hits:
                return best
    assert best is not None, "no point won by the home robot"
    return best


def span(
    pts: list[dict], first: int, cap: float, last: int | None = None
) -> tuple[float, float, int]:
    """Game-time window covering points first..last (inclusive), capped; + points to simulate."""
    last = len(pts) - 1 if last is None else min(last, len(pts) - 1)
    t0 = pts[first]["t_start"] + (0.2 if first > 0 else 0.0)
    return t0, min(pts[last]["t_end"] + 0.8, t0 + cap), last + 2


# ----------------------------------------------------------------------------- results text
def result_rows() -> tuple[list[list[str]], set[int]]:
    df = pd.read_csv(EVAL / "matches.csv")
    order = ["A", "B", "B+", "B*", "C", "D-", "D"]
    rows, score = [], []
    for name, g in sorted(df.groupby("controller"),
                          key=lambda kv: order.index(kv[0].split(":")[0]) if kv[0].split(":")[0] in order else 99):  # fmt: skip
        cells = [name]
        for cond in ("nominal", "randomized", "unknown_wind", "unknown_court"):
            v = g[g["condition"] == cond]["point_win_rate"].mean()
            cells.append(f"{v * 100:.0f}%")
        rows.append(cells)
        score.append(g["point_win_rate"].mean())
    return rows, ({int(np.argmax(score))} if score else set())


# ----------------------------------------------------------------------------- main
def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--out", default=str(OUT))
    a = p.parse_args(argv)
    CLIPS.mkdir(parents=True, exist_ok=True)
    segs: list[Path] = []

    # 1) Opening highlight: the longest rally won by the Hybrid AI
    opp, seed, pt = best_rally(
        HYBRID, ("steady", "random", "lob"), range(20000, 20006), "nominal", 8
    )
    t0, t1 = pt["t_start"] + (0.2 if pt["index"] > 0 else 0.0), pt["t_end"] + 0.8
    t1 = min(t1, t0 + 11.0)
    hl = Clip("01_highlight", HYBRID, opp, seed, t0, t1, pt["index"] + 2,
              title="Robot-Arm Tennis AI",
              captions=((t0, "8 自由度のロボットアーム同士が、物理シミュレーション内でテニスをする"),
                        (t0 + 5.0, "AI がコースと球速を選び、物理モデルが「どう打つか」を計算して実行")))  # fmt: skip
    segs.append(render_clip(hl))
    bg = first_frame(segs[-1])

    # 2) Problem statement
    segs.append(card(
        "課題と研究課題", [
            "・ボールは空気抵抗・回転（マグヌス）・バウンドで曲がり、0.6–1 秒で飛んでくる",
            "・ロボットは関節トルクと可動域に制約がある 8 自由度の機械（ガントリー 2 + アーム 6）",
            "・打球の行き先はラケットの向き・速度・打点・時刻の組合せで決まる（3 次元）",
            "・コートの弾み方・風・ボールの重さは、試合ごとに変わる（未知の環境も）",
            "",
            "研究課題: 物理モデルによる予測・計画に、学習（戦術の強化学習・残差学習）と",
            "オンライン適応を組み合わせると、対戦性能と未知環境への頑健性は向上するか？",
        ], 11, "02_problem", bg=bg, subtitle="Physical AI 応用1 最終課題 / OmniCampus: yoichiozaki",
    ))  # fmt: skip

    # 3) Architecture
    segs.append(card("システム構成", [], 8, "03_architecture", image=FIG / "architecture.png",
                     image_box=(40, 120, 1240, 690)))  # fmt: skip

    # 4) Robot mechanism: slow-motion stroke from the side
    pts = find_points("physics", "steady", 30001, "nominal", 2, first="away")
    hit = next((p["home_hits"][0] for p in pts if p["home_hits"]), 2.0)
    segs.append(render_clip(Clip(
        "04_robot", "physics", "steady", 30001, hit - 1.1, hit + 0.45, 2, speed=0.25, first="away",
        camera="home_closeup", title="ロボット機構とスイング（スロー 0.25x）",
        captions=((hit - 1.1, "床ガントリー 2 軸で打点の近くへ移動、6 軸アームの閉形式 IK で打点・面の向きを解く"),
                  (hit - 0.45, "5 次多項式の関節軌道で「指定時刻に、指定の向きと速度で」ラケットが打点を通過")),
    )))  # fmt: skip

    # 5) What the physics layer predicts (side view, slow)
    pts = find_points("physics", "steady", 30002, "nominal", 2, first="away")
    hit = next((p["home_hits"][0] for p in pts if p["home_hits"]), 2.0)
    segs.append(render_clip(Clip(
        "05_prediction", "physics", "steady", 30002, hit - 1.0, hit + 0.9, 2, speed=0.4, first="away",
        camera="broadcast", title="物理予測（スロー 0.4x）",
        captions=((hit - 1.0, "緑: 空気抵抗とバウンドを含む予測軌道 / 橙: 計画した打点（実際のアルゴリズム出力）"),
                  (hit + 0.02, "黄: 打球の予測軌道 / 青: 狙った着地点。打撃モデルの逆解でラケットの向きと速度を決定")),
    )))  # fmt: skip

    # 6) Controller comparison: same opponent, same seed, the first two points
    for i, (spec, cap) in enumerate((
        ("naive", "A: 直線で飛んでくると仮定する素朴な制御 — 重力とバウンドを無視するので打点に間に合わない"),
        ("physics", "B: 物理予測制御 — 飛行とバウンドを予測し、到達可能な打点で打ち返す（狙いは常にセンター）"),
        (HYBRID, "D: Hybrid AI — 物理予測の上で、強化学習が相手を見てコースと球速を選ぶ"),
    )):  # fmt: skip
        pts = find_points(spec, "steady", 30003, "nominal", 4)
        t0, t1, n = span(pts, 0, 7.0, last=1)
        segs.append(render_clip(Clip(f"06_compare_{i}", spec, "steady", 30003, t0, t1, n,
                                     title="同じ相手・同じ初期条件で比較", captions=((t0, cap),))))  # fmt: skip

    # 7) Unknown court: physics only (first points) vs. online adaptation (after a few points)
    pts = find_points("physics", "steady", 30004, "unknown_court", 5)
    t0, t1, n = span(pts, 0, 7.0)
    segs.append(render_clip(Clip(
        "07_unknown_court_0", "physics", "steady", 30004, t0, t1, n, condition="unknown_court",
        title="未知のコート（よく弾み・滑る）",
        captions=((t0, "B: 標準モデルのまま — バウンドが予測より高く速いので、打点がずれてミスが増える"),),
    )))  # fmt: skip
    pts = find_points(HYBRID, "steady", 30004, "unknown_court", 8)
    k = max(range(2, len(pts)), key=lambda j: (len(pts[j]["home_hits"]), -j))
    t0, t1, n = span(pts, k, 9.0)
    segs.append(render_clip(Clip(
        "07_unknown_court_1", HYBRID, "steady", 30004, t0, t1, n, condition="unknown_court",
        title="未知のコート（よく弾み・滑る）",
        captions=((t0, "D: バウンド前後の速度から反発・摩擦をオンライン推定し、予測モデルを更新して対応"),),
        adapt_lines=True,
    )))  # fmt: skip

    # 8) Unknown wind
    pts = find_points(HYBRID, "steady", 30005, "unknown_wind", 8)
    k = max(range(2, len(pts)), key=lambda j: (len(pts[j]["home_hits"]), -j))
    t0, t1, n = span(pts, k, 8.0)
    segs.append(render_clip(Clip(
        "08_unknown_wind", HYBRID, "steady", 30005, t0, t1, n, condition="unknown_wind",
        title="未知の強風（横風 2–2.5 m/s）",
        captions=((t0, "飛行中の減速と横流れから抗力と風を推定（再帰最小二乗）。予測軌道が風で曲がる"),),
        adapt_lines=True,
    )))  # fmt: skip
    # 9) Results (numbers from results/eval_main)
    rows, best = result_rows()
    segs.append(table_card(
        "定量評価: ポイント取得率（相手 5 種 × 12 試合 × 20 ポイント）",
        ["制御器", "標準", "ランダム化", "未知: 強風", "未知: コート"], rows, 10, "09_results_table",
        col_x=[90, 560, 720, 880, 1040], highlight=best,
        footer="出典: results/eval_main/matches.csv（再生成: python scripts/evaluate.py）",
    ))  # fmt: skip
    for name, fig, sec, title in (
        ("10_point_win", "point_win_rate.png", 7, "条件別のポイント取得率（95% 信頼区間）"),
        ("11_drill", "drill_accuracy.png", 7, "学習と適応の効果: 球出しドリルでの打球精度"),
    ):
        segs.append(card(title, [], sec, name, image=FIG / fig, image_box=(60, 120, 1220, 680)))

    # 11) Advanced 1: self-play league
    pts = find_points(LEAGUE_D, LEAGUE_D, 41000, "nominal", 2)
    segs.append(render_clip(Clip(
        "13_adv_ace", LEAGUE_D, LEAGUE_D, 41000, 0.5, pts[0]["t_end"] + 0.4, 1, camera="broadcast",
        speed=0.5, title="発展1: 元のルール", away_name="相手: D（同じ方策）",
        captions=((0.0, "自己対戦が見つけた「誰も返せないサーブ」: ベースライン際で低く弾み、受け手の台座の横を抜ける"),),
    )))  # fmt: skip
    segs.append(card("発展1: 自己対戦リーグ（サービスボックスあり）", [], 9, "14_adv_league",
                     image=FIG / "league_sb_elo.png", image_box=(40, 120, 1240, 660),
                     footer="元のルールでは全世代がサーブで勝つだけで強さが変わらず → 実際のテニスと同じサービスボックスを導入"))  # fmt: skip
    pts = find_points(SB_CHAMP, SB_D0, 41101, "nominal", 6, service_line=1.62)
    p5 = pts[4]
    segs.append(render_clip(Clip(
        "15_adv_rally", SB_CHAMP, SB_D0, 41101, max(p5["t_end"] - 8.0, p5["t_start"] + 0.2),
        p5["t_end"] + 0.6, 6, service_line=1.62, title="発展1: サービスボックス",
        away_name="相手: 世代0（出発点）",
        captions=((0.0, "最終世代は深く速い球で主導権を取り、出発点の世代0から 98% のポイントを奪う"),),
    )))  # fmt: skip

    # 12) Advanced 2: can a VLA play tennis? (clips rendered on Colab: scripts/vla/render_vla_clips.py)
    segs.append(card("発展2: VLA はテニスができるか（惑星テニス）", [], 9, "16_adv_vla",
                     image=FIG / "vla_worlds.png", image_box=(40, 120, 1240, 660),
                     footer="SmolVLA（4.5 億パラメータ）を物理エキスパートの 1,673 本のデモで微調整（Colab A100）"))  # fmt: skip
    for name in ("vla_moon_e2e", "vla_earth_hier", "vla_earth_hier_late", "vla_moon_hier_late"):
        clip = CLIPS / f"{name}.mp4"
        if clip.exists():
            segs.append(clip)
            meta = json.loads(clip.with_suffix(".json").read_text(encoding="utf-8"))
            manifest.append({"segment": name, "type": "simulation (VLA, rendered on Colab)",
                             "seconds": probe(clip)["duration"], **meta})  # fmt: skip
    segs.append(card("VLA に許される「考える時間」は世界の速さで決まる", [], 8, "17_adv_vla_latency",
                     image=FIG / "vla_hier_latency.png", image_box=(40, 120, 1240, 660)))  # fmt: skip

    # 10) Summary
    segs.append(card(
        "まとめと限界", [
            "・物理モデルが「打ち返す」、学習が戦術と補正、オンライン同定が環境の変化を担当",
            "・自己対戦は設計の死角（誰も返せないサーブ）を最初に見つけた。ルールを直すと出発点に",
            "  98% 勝つまで強くなったが、固定の相手に対する強さはほぼ変わらなかった",
            "・VLA 単独は画像からボールを捉えられず、ほぼ当たらない。ボールの位置を直接与えると",
            "  半分ほど当たるが、狙った所へは返せない（打球の精度と行動のつなぎ方が次の壁）",
            "・階層型（狙いは VLA、打つのは物理）は教師並み。考える時間の上限は世界の速さに比例",
            "・限界: シミュレーションのみ、相手はスクリプト型。詳細は説明資料 PDF を参照",
        ], 11, "12_summary", bg=bg,
        footer="コード・設定・結果: リポジトリ robot-arm-tennis/（README, EXPERIMENTS.md）",
    ))  # fmt: skip

    concat(segs, Path(a.out))
    info = probe(a.out)
    (Path(a.out).parent / "video_manifest.json").write_text(
        json.dumps({"video": str(a.out), "probe": info, "segments": manifest}, indent=2,
                   ensure_ascii=False), encoding="utf-8")  # fmt: skip
    print(json.dumps(info, indent=2))


if __name__ == "__main__":
    main()
