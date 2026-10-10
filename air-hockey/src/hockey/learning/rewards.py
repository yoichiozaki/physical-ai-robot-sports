"""Reward function for the learning agent (weights configurable from YAML)."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from hockey.controllers.prediction import PuckPredictor
from hockey.simulation.physics import SimState, StepResult


@dataclass(frozen=True)
class RewardConfig:
    # Sparse outcome terms
    goal_scored: float = 10.0
    goal_conceded: float = -10.0
    # Event terms
    hit: float = 0.3  # robot touches the puck and it leaves towards the opponent (vx > 0.5 m/s)
    hit_on_target: float = 0.7  # ... and its free-flight path ends in the opponent goal
    defense: float = 1.0  # robot touches the puck while it was heading into our goal
    bad_hit: float = -0.5  # robot sends the puck towards its own goal (vx < -0.5 m/s)
    # Potential-based shaping (gamma * phi(s') - phi(s)), does not change the optimal policy
    shaping_gamma: float = 0.99
    approach: float = 0.5  # phi -= approach * |mallet - puck| while the puck is slow in our half
    block_align: float = (
        1.0  # phi -= block_align * |y_mallet - y_cross| while a shot is on our goal
    )
    # Regularisers
    energy: float = -0.002  # * mean squared normalised velocity command
    illegal: float = -0.05  # target outside the workspace (had to be clipped)
    time: float = 0.0  # per control step


class RewardFunction:
    """Computes the per-control-step reward and keeps the small state it needs."""

    def __init__(
        self, cfg: RewardConfig, truth: PuckPredictor, defense_x: float, vmax: tuple[float, float]
    ):
        self.cfg = cfg
        self.truth = truth
        self.defense_x = defense_x
        self.vmax = vmax
        self.threat = False
        self._phi = 0.0
        self._cache: tuple[float, float | None] = (-1.0, None)

    def reset(self, state: SimState) -> None:
        self.threat = False
        self._cache = (-1.0, None)
        self._phi = self._potential(state)

    def _threat_crossing(self, s: SimState) -> float | None:
        if self._cache[0] == s.time:
            return self._cache[1]
        val = None
        if s.puck_vel[0] < -0.3:
            traj = self.truth.predict(s.puck_pos, s.puck_vel, 1.5)
            if traj.end_event == "goal_own":
                hit = traj.first_crossing_x(self.defense_x)
                val = None if hit is None else hit[1]
        self._cache = (s.time, val)
        return val

    def _potential(self, s: SimState) -> float:
        c = self.cfg
        phi = 0.0
        if c.block_align:
            y_cross = self._threat_crossing(s)
            if y_cross is not None:
                phi -= c.block_align * abs(float(s.home_pos[1]) - y_cross)
        if c.approach and s.puck_pos[0] < 0.0 and math.hypot(*s.puck_vel) < 1.0:
            phi -= c.approach * math.hypot(*(s.home_pos - s.puck_pos))
        return phi

    def __call__(
        self,
        before: SimState,
        result: StepResult,
        after: SimState,
        cmd_vel: tuple[float, float],
        clipped: bool,
    ) -> tuple[float, dict[str, float]]:
        c = self.cfg
        terms: dict[str, float] = {}
        if result.goal == "home":
            terms["goal"] = c.goal_scored
        elif result.goal == "away":
            terms["goal"] = c.goal_conceded
        threat_before = self._threat_crossing(before) is not None
        touched = False
        for ev in result.events:
            if ev.kind != "home":
                continue
            touched = True
            vx = float(after.puck_vel[0])
            if vx > 0.5:
                terms["hit"] = terms.get("hit", 0.0) + c.hit
                if c.hit_on_target:
                    traj = self.truth.predict(after.puck_pos, after.puck_vel, 2.0)
                    if traj.end_event == "goal_opp":
                        terms["hit_on_target"] = terms.get("hit_on_target", 0.0) + c.hit_on_target
            elif vx < -0.5:
                terms["bad_hit"] = terms.get("bad_hit", 0.0) + c.bad_hit
        if touched and (threat_before or self.threat) and result.goal != "away":
            terms["defense"] = c.defense
            self.threat = False
        elif threat_before:
            self.threat = True
        if not touched and after.puck_vel[0] > 0.05:
            self.threat = False
        if result.goal is None:
            phi = self._potential(after)
            terms["shaping"] = c.shaping_gamma * phi - self._phi
            self._phi = phi
        else:
            terms["shaping"] = -self._phi  # terminal: phi(s_T) = 0
            self._phi = 0.0
        if c.energy:
            terms["energy"] = (
                c.energy * 0.5 * sum((u / v) ** 2 for u, v in zip(cmd_vel, self.vmax, strict=True))
            )
        if clipped and c.illegal:
            terms["illegal"] = c.illegal
        if c.time:
            terms["time"] = c.time
        return float(np.sum(list(terms.values()))), terms
