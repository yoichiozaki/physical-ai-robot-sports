"""Planet worlds: change gravity to change how fast the game is, nothing else.

With gravity g' = s^2 g and every launch speed scaled by s, a ballistic path keeps exactly
its shape and only its time is stretched by 1/s: gravity scales with s^2 and so do the
aerodynamic (drag ~ v^2, Magnus ~ w v) accelerations. The robots are the same machines in
every world (same motors), so a slower world gives every controller - and a slow-thinking
VLA - proportionally more time.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace

from tennis.control.ballistics import BallModel, G
from tennis.control.players import PhysicsPlayer, PlayerConfig
from tennis.sim.config import TennisConfig


@dataclass(frozen=True)
class World:
    name: str
    label_ja: str
    gravity: float

    @property
    def time_scale(self) -> float:
        """s: speeds scale by s, durations by 1/s (Earth = 1)."""
        return math.sqrt(self.gravity / G)


WORLDS = {
    "earth": World("earth", "地球", 9.81),
    "mars": World("mars", "火星", 3.71),
    "moon": World("moon", "月", 1.62),
}


def sim_config(world: World, base: TennisConfig | None = None) -> TennisConfig:
    base = base or TennisConfig()
    return replace(base, physics=replace(base.physics, gravity=world.gravity))


def ball_model(world: World) -> BallModel:
    return BallModel(gravity=world.gravity)


def player_config(world: World, name: str = "Expert") -> PlayerConfig:
    return PlayerConfig(name=name, horizon=2.2 / world.time_scale)


def make_expert(world: World, cfg: TennisConfig, strategy=None, name: str = "Expert"):
    return PhysicsPlayer(cfg, player_config(world, name), strategy, ball_model(world))
