"""Scripted opponent strategies (all run in their own player frame)."""

from __future__ import annotations

from collections import deque
from dataclasses import replace

import numpy as np

from hockey.controllers.base import Command, Controller, DecisionInfo
from hockey.controllers.physics_based import PhysicsController, PhysicsControllerConfig
from hockey.controllers.prediction import PuckModel
from hockey.controllers.reactive import ReactiveController
from hockey.observation import Geometry, PlayerView

OPPONENT_NAMES = ("stationary", "random", "reactive", "defensive", "aggressive")


class StationaryOpponent:
    """Mallet stays at its home position (a static goalkeeper)."""

    name = "Stationary"

    def __init__(self, geom: Geometry, y: float = 0.0) -> None:
        self.target = np.array([geom.home_x, y])

    def reset(self, rng: np.random.Generator) -> None:
        return None

    def act(self, view: PlayerView) -> Command:
        return Command(self.target.copy(), DecisionInfo(mode="hold", strategy="Stationary"))


class RandomOpponent:
    """Moves to random points of its half, re-sampled every 0.3-0.8 s."""

    name = "Random"

    def __init__(self, geom: Geometry) -> None:
        self.geom = geom
        self.rng = np.random.default_rng(0)
        self.target = np.array([geom.home_x, 0.0])
        self.next_change = 0.0

    def reset(self, rng: np.random.Generator) -> None:
        self.rng = rng
        self.next_change = 0.0

    def act(self, view: PlayerView) -> Command:
        if view.time >= self.next_change:
            g = self.geom
            self.target = np.array(
                [self.rng.uniform(g.x_range[0], g.x_range[1] - 0.1), self.rng.uniform(*g.y_range)]
            )
            self.next_change = view.time + self.rng.uniform(0.3, 0.8)
        return Command(self.target.copy(), DecisionInfo(mode="random", strategy="Random"))


class DelayedController:
    """Wraps a controller with a perception delay (human-like reaction time).

    The wrapped controller sees the puck/opponent state from ``delay`` seconds ago (its own
    encoder state stays current), so it reacts late to fast shots and wall bounces.
    """

    def __init__(
        self, inner: Controller, delay: float, control_dt: float, speed_factor: float = 1.0
    ) -> None:
        self.inner = inner
        self.name = inner.name
        self.steps = max(0, round(delay / control_dt))
        self._buf: deque[PlayerView] = deque(maxlen=self.steps + 1)
        self.speed_factor = speed_factor  # inherent robot speed of this opponent (x condition DR)

    def reset(self, rng: np.random.Generator) -> None:
        self._buf.clear()
        self.inner.reset(rng)

    def set_speed_scale(self, scale: float) -> None:
        if hasattr(self.inner, "set_speed_scale"):
            self.inner.set_speed_scale(scale)

    def act(self, view: PlayerView) -> Command:
        self._buf.append(view)
        old = self._buf[0]
        delayed = replace(
            view,
            puck_pos=old.puck_pos,
            puck_vel=old.puck_vel,
            opp_pos=old.opp_pos,
            opp_vel=old.opp_vel,
        )
        return self.inner.act(delayed)


DEFENSIVE_CFG = PhysicsControllerConfig(
    name="Defensive",
    attack_moving=False,
    attack_slow=True,
    shot_speed=2.5,
    defense_offset=0.10,
    retreat_offset=0.10,
    retreat_y_gain=0.8,
)

AGGRESSIVE_CFG = PhysicsControllerConfig(
    name="Aggressive",
    attack_moving=True,
    attack_slow=True,
    shot_speed=4.5,
    aim_jitter=0.09,
    retreat_offset=0.45,
    retreat_y_gain=0.6,
)

# Imperfect opponents: reaction delay [s], robot speed factor and a slightly wrong physical
# model (they do not know the table's exact restitution/friction), like human players.
OPPONENT_DELAY = {"reactive": 0.08, "defensive": 0.18, "aggressive": 0.12, "physics": 0.10}
OPPONENT_SPEED = {"defensive": 0.8}
OPPONENT_MODEL = PuckModel(decay=0.10, wall_restitution=0.92, mallet_restitution=0.70)


def make_opponent(
    name: str, geom: Geometry, model: PuckModel | None = None, imperfect: bool = True
) -> Controller:
    name = name.lower()
    opp_model = OPPONENT_MODEL if imperfect else model
    if name == "stationary":
        return StationaryOpponent(geom)
    if name == "random":
        return RandomOpponent(geom)
    if name == "reactive":
        ctrl: Controller = ReactiveController(geom)
    elif name == "defensive":
        ctrl = PhysicsController(geom, DEFENSIVE_CFG, opp_model)
    elif name == "aggressive":
        ctrl = PhysicsController(geom, AGGRESSIVE_CFG, opp_model)
    elif name == "physics":
        ctrl = PhysicsController(geom, PhysicsControllerConfig(name="PhysicsOpp"), opp_model)
    else:
        raise ValueError(f"Unknown opponent '{name}'. Choose from {(*OPPONENT_NAMES, 'physics')}")
    if imperfect and OPPONENT_DELAY.get(name, 0.0) > 0:
        return DelayedController(
            ctrl, OPPONENT_DELAY[name], geom.control_dt, OPPONENT_SPEED.get(name, 1.0)
        )
    return ctrl
