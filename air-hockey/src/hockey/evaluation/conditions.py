"""Evaluation / training conditions: domain-randomisation ranges and sensor noise."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from hockey.observation import SensorNoise
from hockey.simulation.physics import PhysicsParams


@dataclass(frozen=True)
class ConditionConfig:
    name: str = "nominal"
    puck_decay: tuple[float, float] = (0.15, 0.15)
    wall_restitution: tuple[float, float] = (0.85, 0.85)
    mallet_restitution: tuple[float, float] = (0.70, 0.70)
    opponent_speed: tuple[float, float] = (1.0, 1.0)
    obs_noise_pos: float = 0.0
    obs_noise_vel: float = 0.0

    def sample(self, rng: np.random.Generator) -> tuple[PhysicsParams, SensorNoise]:
        u = lambda r: float(rng.uniform(r[0], r[1])) if r[1] > r[0] else float(r[0])  # noqa: E731
        params = PhysicsParams(
            puck_decay=u(self.puck_decay),
            wall_restitution=u(self.wall_restitution),
            mallet_restitution=u(self.mallet_restitution),
            home_speed_scale=1.0,
            away_speed_scale=u(self.opponent_speed),
        )
        return params, SensorNoise(self.obs_noise_pos, self.obs_noise_vel)


NOMINAL = ConditionConfig()

# Training distribution of the learned controllers (domain randomisation).
RANDOMIZED = ConditionConfig(
    name="randomized",
    puck_decay=(0.05, 0.35),
    wall_restitution=(0.75, 0.92),
    mallet_restitution=(0.60, 0.80),
    opponent_speed=(0.8, 1.1),
    obs_noise_pos=0.003,
    obs_noise_vel=0.03,
)

# Held-out conditions never seen during training (generalisation tests).
UNKNOWN_STICKY = ConditionConfig(
    name="unknown_sticky",
    puck_decay=(0.45, 0.65),
    wall_restitution=(0.62, 0.70),
    mallet_restitution=(0.52, 0.58),
    opponent_speed=(1.15, 1.30),
    obs_noise_pos=0.006,
    obs_noise_vel=0.08,
)
UNKNOWN_BOUNCY = ConditionConfig(
    name="unknown_bouncy",
    puck_decay=(0.0, 0.03),
    wall_restitution=(0.94, 0.98),
    mallet_restitution=(0.82, 0.88),
    opponent_speed=(1.15, 1.30),
    obs_noise_pos=0.006,
    obs_noise_vel=0.08,
)

CONDITIONS = {c.name: c for c in (NOMINAL, RANDOMIZED, UNKNOWN_STICKY, UNKNOWN_BOUNCY)}
