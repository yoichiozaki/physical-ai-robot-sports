"""Baseline A: reactive controller that follows the *current* puck position (no prediction)."""

from __future__ import annotations

import numpy as np

from hockey.controllers.base import Command, DecisionInfo
from hockey.observation import Geometry, PlayerView


class ReactiveController:
    """Chase the puck in our half (pushing it towards +x); otherwise mirror its y at home.

    Args:
        push: how far beyond the puck centre the mallet aims (makes it hit through the puck).
        track_gain: fraction of the puck's y followed while guarding.
    """

    name = "Reactive"

    def __init__(
        self,
        geom: Geometry,
        push: float = 0.10,
        track_gain: float = 1.0,
        attack_line: float = -0.02,
    ):
        self.geom = geom
        self.push = push
        self.track_gain = track_gain
        self.attack_line = attack_line

    def reset(self, rng: np.random.Generator) -> None:
        return None

    def act(self, view: PlayerView) -> Command:
        g = self.geom
        px, py = view.puck_pos
        if px < self.attack_line:
            target = np.array([px + self.push, py])
            mode = "chase"
        else:
            goal_half = g.goal_width / 2 + g.mallet_radius
            target = np.array(
                [g.home_x, float(np.clip(self.track_gain * py, -goal_half, goal_half))]
            )
            mode = "track"
        return Command(target=target, info=DecisionInfo(mode=mode, strategy="Follow puck"))
