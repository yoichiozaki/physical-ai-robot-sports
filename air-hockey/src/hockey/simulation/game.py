"""Game rules: score keeping, face-offs, possession timeout, match clock."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from hockey.simulation.physics import HockeySim, StepResult


@dataclass(frozen=True)
class GameConfig:
    match_duration: float = 60.0  # game-time seconds per match
    possession_timeout: float = 7.0  # puck idle in one half -> face-off to the other player
    faceoff_x: float = 0.45  # |x| of the face-off spot
    faceoff_y: float = 0.20  # face-off y is uniform in [-faceoff_y, faceoff_y]


class Referee:
    """Applies the rules on top of ``HockeySim``. Scores are (home, away)."""

    def __init__(
        self, sim: HockeySim, cfg: GameConfig | None = None, rng: np.random.Generator | None = None
    ):
        self.sim = sim
        self.cfg = cfg or GameConfig()
        self.rng = rng or np.random.default_rng(0)
        self.score = [0, 0]
        self.points: list[dict] = []
        self.anomalies = 0
        self.possession_resets = 0
        self._side_since = 0.0
        self._side = 0
        self._point_start = 0.0

    @property
    def time(self) -> float:
        return self.sim.time

    @property
    def time_remaining(self) -> float:
        return max(0.0, self.cfg.match_duration - self.sim.time)

    @property
    def done(self) -> bool:
        return self.sim.time >= self.cfg.match_duration - 1e-9

    def start(self, first: str | None = None) -> None:
        self.score = [0, 0]
        self.points = []
        self.anomalies = 0
        self.possession_resets = 0
        side = first or ("home" if self.rng.random() < 0.5 else "away")
        self.faceoff(side, keep_time=False)

    def faceoff(self, side: str, keep_time: bool = True) -> None:
        sx = -1.0 if side == "home" else 1.0
        y = self.rng.uniform(-self.cfg.faceoff_y, self.cfg.faceoff_y)
        self.sim.reset(
            puck_pos=(sx * self.cfg.faceoff_x, y), puck_vel=(0.0, 0.0), keep_time=keep_time
        )
        self._side = -1 if side == "home" else 1
        self._side_since = self.sim.time
        self._point_start = self.sim.time

    def update(self, result: StepResult) -> str | None:
        """Apply the rules after a control step.

        Returns 'goal_home' / 'goal_away' (scorer), 'out_of_bounds', 'possession_home' /
        'possession_away' (timeout face-off) or None.
        """
        if result.goal is not None:
            scorer = result.goal
            self.score[0 if scorer == "home" else 1] += 1
            self.points.append({"scorer": scorer, "start": self._point_start, "end": self.sim.time})
            self.faceoff("away" if scorer == "home" else "home")
            return f"goal_{scorer}"
        if result.out_of_bounds:
            self.anomalies += 1
            self.faceoff("home" if self.rng.random() < 0.5 else "away")
            return "out_of_bounds"
        px = float(self.sim.data.qpos[self.sim._puck_q[0]])
        side = -1 if px < 0 else 1
        if side != self._side:
            self._side = side
            self._side_since = self.sim.time
        elif self.sim.time - self._side_since > self.cfg.possession_timeout:
            self.possession_resets += 1
            holder = "home" if side < 0 else "away"
            self.faceoff("away" if holder == "home" else "home")
            return f"possession_{holder}"
        return None
