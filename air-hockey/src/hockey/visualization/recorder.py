"""Turn simulation steps into annotated video frames (MuJoCo render + overlays + HUD)."""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from hockey.evaluation.match import StepRecord
from hockey.visualization.hud import HudState, draw_hud
from hockey.visualization.renderer import SceneRenderer, decision_overlays
from hockey.visualization.video import VideoWriter

STRATEGY_JA = {
    "Defense": "防御",
    "Direct shot": "ストレートシュート",
    "Bank shot (L)": "バンクショット（左壁）",
    "Bank shot (R)": "バンクショット（右壁）",
    "Follow puck": "パック追従",
    "RL target": "強化学習（目標位置を直接出力）",
    "Stationary": "静止",
    "Random": "ランダム",
}
MODE_JA = {
    "strike": "打撃",
    "position": "打撃準備（助走位置へ移動）",
    "defend": "防御（予測地点で迎撃）",
    "guard": "待機（ゴール前）",
    "chase": "パックを追う",
    "track": "ゴール前で横移動",
    "policy": "方策の出力位置へ移動",
}


def speed_label(speed: float) -> str:
    if abs(speed - 1.0) < 1e-6:
        return "再生速度 1.0x（実時間）"
    if speed < 1.0:
        return f"スロー再生 {speed:.2g}x"
    return f"{speed:.2g}倍速"


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
    """Callable for ``play_match(on_step=...)``: renders frames at the video frame rate."""

    def __init__(
        self,
        writer: VideoWriter | None,
        cfg: RecorderConfig,
        caption_fn: Callable[[float], str] | None = None,
    ) -> None:
        self.writer = writer
        self.cfg = cfg
        self.caption_fn = caption_fn
        self.renderer: SceneRenderer | None = None
        self.next_t: float | None = None
        self.t0: float | None = None
        self.log: list[dict] = []
        self.frames: list[np.ndarray] = []
        self.keep_frames = writer is None

    def close(self) -> None:
        if self.renderer is not None:
            self.renderer.close()
            self.renderer = None

    def __call__(self, rec: StepRecord) -> None:
        sim = rec.sim
        if self.renderer is None:
            self.renderer = SceneRenderer(sim, self.cfg.width, self.cfg.height, self.cfg.camera)
        st = sim.state()
        if self.t0 is None:
            self.t0 = st.time - sim.control_dt
            self.next_t = self.t0
        info = rec.home_cmd.info
        self.log.append(
            {
                "t": round(st.time, 4),
                "puck": [round(float(v), 4) for v in (*st.puck_pos, *st.puck_vel)],
                "home": [round(float(v), 4) for v in st.home_pos],
                "away": [round(float(v), 4) for v in st.away_pos],
                "home_target": [round(float(v), 4) for v in rec.home_cmd.target],
                "mode": info.mode,
                "strategy": info.strategy,
                "intercept": None
                if info.intercept is None
                else [round(v, 4) for v in info.intercept],
                "events": [e.kind for e in rec.result.events],
                "referee": rec.event,
                "score": list(rec.score),
            }
        )
        assert self.next_t is not None
        while self.next_t <= st.time + 1e-9:
            frame = self.frame(rec)
            if self.writer is not None:
                self.writer.write(frame)
            if self.keep_frames:
                self.frames.append(frame)
            self.next_t += self.cfg.speed / self.cfg.fps

    def frame(self, rec: StepRecord) -> np.ndarray:
        cfg = self.cfg
        geom_r = (rec.sim.cfg.puck.radius, rec.sim.cfg.mallet.radius)
        overlays = []
        if cfg.overlays_home:
            overlays += decision_overlays(rec.home_cmd.info, "home", rec.home_cmd.target, *geom_r)
        if cfg.overlays_away:
            overlays += decision_overlays(rec.away_cmd.info, "away", rec.away_cmd.target, *geom_r)
        assert self.renderer is not None
        img = self.renderer.render(overlays)
        info = rec.home_cmd.info
        lines = list(cfg.extra_lines)
        if cfg.show_decision and info.intercept_time is not None:
            lines.append(f"到達予測|{info.intercept_time:.2f} 秒後")
        if (
            cfg.show_decision
            and "mallet_speed" in info.extra
            and info.mode in ("strike", "position")
        ):
            lines.append(f"打撃速度|{info.extra['mallet_speed']:.1f} m/s")
        t_elapsed = rec.sim.time - (self.t0 or 0.0)
        caption = self.caption_fn(t_elapsed) if self.caption_fn else cfg.caption
        hud = HudState(
            home_name=cfg.home_name,
            away_name=cfg.away_name,
            score=rec.score,
            time_text=f"試合時間 {rec.sim.time:5.1f} s",
            strategy=STRATEGY_JA.get(info.strategy, info.strategy) if cfg.show_decision else "",
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
