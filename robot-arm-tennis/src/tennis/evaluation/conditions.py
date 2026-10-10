"""Evaluation / training conditions: domain-randomisation ranges and sensor noise."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from tennis.sim.physics import PhysicsParams


@dataclass(frozen=True)
class ConditionConfig:
    name: str = "nominal"
    court_restitution: tuple[float, float] = (0.75, 0.75)
    court_friction: tuple[float, float] = (0.6, 0.6)
    racket_restitution: tuple[float, float] = (0.85, 0.85)
    drag_scale: tuple[float, float] = (1.0, 1.0)
    wind_speed: tuple[float, float] = (
        0.0,
        0.0,
    )  # horizontal wind magnitude [m/s], random direction
    wind_dir: float | None = None  # fixed direction [rad] (None: random)
    opponent_speed: tuple[float, float] = (1.0, 1.0)
    noise_pos: float = 0.0
    noise_vel: float = 0.0

    def sample(self, rng: np.random.Generator) -> tuple[PhysicsParams, float, float]:
        u = lambda r: float(rng.uniform(r[0], r[1])) if r[1] > r[0] else float(r[0])  # noqa: E731
        ws = u(self.wind_speed)
        wd = float(rng.uniform(0, 2 * np.pi)) if self.wind_dir is None else self.wind_dir
        params = PhysicsParams(
            court_restitution=u(self.court_restitution),
            court_friction=u(self.court_friction),
            racket_restitution=u(self.racket_restitution),
            drag_scale=u(self.drag_scale),
            wind=(ws * np.cos(wd), ws * np.sin(wd), 0.0),
            speed_scale=(1.0, u(self.opponent_speed)),
        )
        return params, self.noise_pos, self.noise_vel


NOMINAL = ConditionConfig()
RANDOMIZED = ConditionConfig(
    name="randomized",
    court_restitution=(0.68, 0.82),
    court_friction=(0.4, 0.8),
    racket_restitution=(0.78, 0.9),
    drag_scale=(0.75, 1.3),
    wind_speed=(0.0, 0.8),
    opponent_speed=(0.9, 1.1),
    noise_pos=0.004,
    noise_vel=0.05,
)
# Held-out conditions (never used for training): strong cross-wind + heavy ball, and a
# fast slippery court with lively rackets and faster opponents.
UNKNOWN_WIND = ConditionConfig(
    name="unknown_wind",
    drag_scale=(1.5, 1.7),
    wind_speed=(2.0, 2.5),
    wind_dir=np.pi / 2,
    noise_pos=0.008,
    noise_vel=0.10,
)
UNKNOWN_COURT = ConditionConfig(
    name="unknown_court",
    court_restitution=(0.84, 0.88),
    court_friction=(0.15, 0.25),
    racket_restitution=(0.92, 0.95),
    opponent_speed=(1.15, 1.25),
    noise_pos=0.008,
    noise_vel=0.10,
)
CONDITIONS = {c.name: c for c in (NOMINAL, RANDOMIZED, UNKNOWN_WIND, UNKNOWN_COURT)}
