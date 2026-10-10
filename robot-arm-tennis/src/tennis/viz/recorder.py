"""Turn simulation steps into annotated video frames (MuJoCo render + overlays + HUD)."""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from tennis.evaluation.match import StepRecord
from tennis.viz.hud import HudState, draw_hud
from tennis.viz.renderer import SceneRenderer, decision_overlays
from tennis.viz.video import VideoWriter

MODE_JA = {
    "ready": "待機（構え）",
    "chase": "打点へ移動（ガントリー）",
    "prepare": "打撃準備（軌道計画）",
    "swing": "スイング",
    "follow": "フォロースルー",
    "recover": "定位置へ戻る",
    "let": "見送り（アウトと予測）",
}
DEPTH_JA = {"0": "浅く", "1": "中間の深さ", "2": "深く"}  # rl-{depth}{side}{speed}
SIDE_JA = {"0": "右へ", "1": "センターへ", "2": "左へ"}  # seen from behind the home robot
SPEED_JA = {"0": "遅い球", "1": "中速", "2": "速い球"}


def strategy_ja(label: str, target: tuple[float, float] | None = None) -> str:
    if label.startswith("rl-direct"):
        return "強化学習（ラケット角度・速度を直接出力）"
    if label.startswith("rl-") and len(label) == 6:
        ix, iy, iv = label[3], label[4], label[5]
        return f"強化学習: {DEPTH_JA[ix]}・{SIDE_JA[iy]}・{SPEED_JA[iv]}"
    return {
        "centre": "センターへ返球",
        "open-court": "オープンコート狙い（ルール）",
        "fixed stroke": "固定ストローク",
        "random": "ランダム",
        "corner": "コーナー狙い",
        "lob": "ロブ",
        "fallback": "センターへ返球",
    }.get(label, label)


def speed_label(speed: float) -> str:
    if abs(speed - 1.0) < 1e-6:
        return "再生速度 1.0x（実時間）"
    if speed < 1.0:
        return f"スロー再生 {speed:.2g}x"
    return f"{speed:.2g}倍速"


def adaptation_lines(player) -> list[str]:
    """HUD lines with the online-identified physics of an adaptive player."""
    ma = getattr(player, "model_adapter", None)
    if ma is None or ma._base is None:
        return []
    m0 = ma._base
    wind = float(np.hypot(*ma.wind))
    lines = [
        f"推定 反発係数|{ma.e:.2f}（標準 {m0.bounce_e:.2f}）",
        f"推定 摩擦係数|{ma.mu:.2f}（標準 {m0.bounce_mu:.2f}）",
        f"推定 空気抵抗|{ma.k_drag / m0.k_drag:.2f} 倍",
        f"推定 風|{wind:.1f} m/s",
    ]
    state = "未知の環境 → 推定値で予測" if ma.shifted else "標準モデルで予測"
    lines.append(f"適応|{state}")
    return lines


@dataclass
class RecorderConfig:
    width: int = 1280
    height: int = 720
    fps: int = 30
    speed: float = 1.0
    camera: str = "broadcast"
    home_name: str = "AI"
    away_name: str = "Opponent"
    overlays_home: bool = True
    overlays_away: bool = False
    title: str = ""
    caption: str = ""
    legend: bool = True
    show_decision: bool = True
    extra_lines: list[str] = field(default_factory=list)


class ClipRecorder:
    """Callable for ``Match(on_step=...)``: renders frames at the video frame rate."""

    def __init__(self, writer: VideoWriter | None, cfg: RecorderConfig,
                 caption_fn: Callable[[float], str] | None = None,
                 lines_fn: Callable[[], list[str]] | None = None) -> None:  # fmt: skip
        self.writer = writer
        self.cfg = cfg
        self.caption_fn = caption_fn
        self.lines_fn = lines_fn
        self.renderer: SceneRenderer | None = None
        self.next_t: float | None = None
        self.t0: float | None = None
        self.log: list[dict] = []
        self.frames: list[np.ndarray] = []
        self.keep_frames = writer is None
        self.enabled = True

    def close(self) -> None:
        if self.renderer is not None:
            self.renderer.close()
            self.renderer = None

    def __call__(self, rec: StepRecord) -> None:
        if not self.enabled:
            return
        sim = rec.sim
        if self.renderer is None:
            self.renderer = SceneRenderer(sim, self.cfg.width, self.cfg.height, self.cfg.camera)
        t = sim.time
        if self.t0 is None:
            self.t0 = t
            self.next_t = t
        info = rec.infos["home"]
        self.log.append(
            {
                "t": round(t, 4),
                "ball": [round(float(v), 4) for v in (*rec.state.ball_pos, *rec.state.ball_vel)],
                "mode": info.mode,
                "strategy": info.strategy,
                "contact": None if info.contact is None else [round(v, 4) for v in info.contact],
                "events": [e.kind for e in rec.result.events],
                "point": None if rec.point is None else rec.point.reason,
                "score": [rec.referee.score["home"], rec.referee.score["away"]],
            }
        )
        assert self.next_t is not None
        while self.next_t <= t + 1e-9:
            frame = self.frame(rec)
            if self.writer is not None:
                self.writer.write(frame)
            if self.keep_frames:
                self.frames.append(frame)
            self.next_t += self.cfg.speed / self.cfg.fps

    def frame(self, rec: StepRecord) -> np.ndarray:
        cfg = self.cfg
        overlays = []
        if cfg.overlays_home:
            overlays += decision_overlays(rec.infos["home"], "home")
        if cfg.overlays_away:
            overlays += decision_overlays(rec.infos["away"], "away")
        assert self.renderer is not None
        img = self.renderer.render(overlays)
        info = rec.infos["home"]
        lines = list(cfg.extra_lines)
        if cfg.show_decision and info.contact_time is not None and info.contact_time > 0:
            lines.append(f"打点まで|{info.contact_time:.2f} 秒")
        if cfg.show_decision and "racket_speed" in info.extra and info.mode in ("prepare", "swing"):
            lines.append(f"ラケット速度|{info.extra['racket_speed']:.1f} m/s")
        if self.lines_fn is not None:
            lines += self.lines_fn()
        t_elapsed = rec.sim.time - (self.t0 or 0.0)
        caption = self.caption_fn(t_elapsed) if self.caption_fn else cfg.caption
        ref = rec.referee
        hud = HudState(
            home_name=cfg.home_name,
            away_name=cfg.away_name,
            score=(ref.score["home"], ref.score["away"]),
            time_text=f"試合時間 {rec.sim.time:5.1f} s",
            strategy=strategy_ja(info.strategy, info.target) if cfg.show_decision else "",
            mode=MODE_JA.get(info.mode, info.mode) if cfg.show_decision else "",
            lines=lines,
            caption=caption,
            speed_label=speed_label(cfg.speed),
            title=cfg.title,
            legend=cfg.legend and cfg.overlays_home,
        )
        return draw_hud(img, hud)

    def save_log(self, path: str | Path, meta: dict | None = None) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(
            json.dumps({"meta": meta or {}, "steps": self.log}, ensure_ascii=False),
            encoding="utf-8",
        )
