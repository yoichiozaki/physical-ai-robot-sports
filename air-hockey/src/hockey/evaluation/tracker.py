"""Per-match metric tracking (home-player perspective)."""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field

import numpy as np

from hockey.controllers.base import DecisionInfo
from hockey.controllers.prediction import FORECAST_HORIZON, PuckModel, PuckPredictor
from hockey.observation import Geometry
from hockey.simulation.physics import SimState, StepResult


@dataclass
class MatchResult:
    goals_for: int = 0
    goals_against: int = 0
    duration: float = 0.0
    threats: int = 0
    saves: int = 0
    hits_home: int = 0
    hits_away: int = 0
    rally_hits: list[int] = field(default_factory=list)
    latencies_ms: list[float] = field(default_factory=list)
    prediction_errors: list[float] = field(default_factory=list)
    intercept_errors: list[float] = field(default_factory=list)
    own_goals: int = 0
    anomalies: int = 0
    possession_resets: int = 0
    strategy_counts: dict[str, int] = field(default_factory=dict)

    @property
    def outcome(self) -> str:
        if self.goals_for > self.goals_against:
            return "win"
        if self.goals_for < self.goals_against:
            return "loss"
        return "draw"

    def summary(self) -> dict[str, float | int | str]:
        lat = np.array(self.latencies_ms) if self.latencies_ms else np.array([np.nan])
        return {
            "goals_for": self.goals_for,
            "goals_against": self.goals_against,
            "goal_diff": self.goals_for - self.goals_against,
            "outcome": self.outcome,
            "win": int(self.outcome == "win"),
            "draw": int(self.outcome == "draw"),
            "loss": int(self.outcome == "loss"),
            "threats": self.threats,
            "saves": self.saves,
            "defense_success_rate": self.saves / self.threats if self.threats else float("nan"),
            "hits_home": self.hits_home,
            "hits_away": self.hits_away,
            "avg_rally_hits": float(np.mean(self.rally_hits)) if self.rally_hits else float("nan"),
            "latency_ms_mean": float(np.nanmean(lat)),
            "latency_ms_p95": float(np.nanpercentile(lat, 95)),
            "prediction_error_m": float(np.mean(self.prediction_errors))
            if self.prediction_errors
            else float("nan"),
            "intercept_error_m": float(np.mean(self.intercept_errors))
            if self.intercept_errors
            else float("nan"),
            "own_goals": self.own_goals,
            "anomalies": self.anomalies,
            "possession_resets": self.possession_resets,
            "duration_s": self.duration,
            **{f"strategy_{k}": v for k, v in sorted(self.strategy_counts.items())},
        }


class MatchTracker:
    """Consumes (state, step result, decision) per control step and accumulates metrics.

    * Threat: puck moving towards the home goal (vx < -0.3 m/s) whose *true* free-flight
      path (simulator parameters, mallets ignored) ends in the home goal within 1.5 s.
      A threat ends as a save if the home mallet touches the puck and no goal follows
      within 1 s; as a failure if a goal is conceded; otherwise it is discarded.
    * Prediction error: the home controller's own forecast of the puck position 0.5 s
      ahead vs. the realised position, over windows without any mallet contact.
    * Intercept error: planned vs. realised puck position at the home robot's impact.
    """

    def __init__(self, geom: Geometry, true_model: PuckModel, control_dt: float) -> None:
        self.geom = geom
        self.truth = PuckPredictor(geom, true_model)
        self.dt = control_dt
        self.r = MatchResult()
        self._threat_active = False
        self._threat_touched_at: float | None = None
        self._rally = 0
        self._pending_pred: deque[tuple[float, float, float]] = deque()
        self._last_contact_time = -1.0
        self._last_info: DecisionInfo | None = None
        self._last_home_touch = -10.0

    def update(
        self,
        before: SimState,
        result: StepResult,
        after: SimState,
        home_info: DecisionInfo | None,
        latency_s: float,
        referee_event: str | None,
    ) -> None:
        r = self.r
        r.latencies_ms.append(latency_s * 1e3)
        if home_info is not None and home_info.strategy:
            r.strategy_counts[home_info.strategy] = r.strategy_counts.get(home_info.strategy, 0) + 1
        t = after.time
        if result.mallet_contact:
            self._last_contact_time = t
        # Contacts
        for ev in result.events:
            if ev.kind == "home":
                r.hits_home += 1
                self._rally += 1
                self._last_home_touch = ev.time
                self._last_contact_time = ev.time
                info = self._last_info
                if info is not None and info.mode == "strike" and info.intercept is not None:
                    r.intercept_errors.append(math.dist(info.intercept, ev.puck_pos))
                if self._threat_active and self._threat_touched_at is None:
                    self._threat_touched_at = ev.time
            elif ev.kind == "away":
                r.hits_away += 1
                self._rally += 1
                self._last_contact_time = ev.time
        # Goals / threats
        if referee_event in ("goal_home", "goal_away"):
            if referee_event == "goal_home":
                r.goals_for += 1
            else:
                r.goals_against += 1
                if self._threat_active:
                    r.threats += 1  # counted as failure
                if t - self._last_home_touch < 0.6 and before.puck_vel[0] < 0:
                    r.own_goals += int(self._last_contact_time == self._last_home_touch)
            r.rally_hits.append(self._rally)
            self._rally = 0
            self._threat_active = False
            self._threat_touched_at = None
            self._pending_pred.clear()
            self._last_info = home_info
            return
        if referee_event is not None:  # face-off without goal
            self._threat_active = False
            self._threat_touched_at = None
            self._pending_pred.clear()
        else:
            self._update_threat(after)
            self._update_prediction(before, after, home_info)
        self._last_info = home_info

    def _update_threat(self, s: SimState) -> None:
        if self._threat_active:
            if self._threat_touched_at is not None and s.time - self._threat_touched_at > 1.0:
                self.r.threats += 1
                self.r.saves += 1
                self._threat_active = False
                self._threat_touched_at = None
            elif self._threat_touched_at is None and s.puck_vel[0] > 0.05:
                self._threat_active = False  # deflected without our touch: discard
            return
        if s.puck_vel[0] < -0.3:
            traj = self.truth.predict(s.puck_pos, s.puck_vel, 1.5)
            if traj.end_event == "goal_own":
                self._threat_active = True
                self._threat_touched_at = None

    def _update_prediction(self, before: SimState, s: SimState, info: DecisionInfo | None) -> None:
        """Forecasts are issued from ``before`` (the state the controller saw), checked on ``s``."""
        while self._pending_pred and self._pending_pred[0][0] <= s.time + 1e-6:
            t_due, px, py = self._pending_pred.popleft()
            if (
                abs(t_due - s.time) < 1e-6
                and self._last_contact_time < t_due - FORECAST_HORIZON - 1e-9
            ):
                self.r.prediction_errors.append(math.hypot(px - s.puck_pos[0], py - s.puck_pos[1]))
        if info is None or "forecast_x" not in info.extra:
            return
        if float(np.hypot(*before.puck_vel)) > 0.3:
            self._pending_pred.append(
                (before.time + FORECAST_HORIZON, info.extra["forecast_x"], info.extra["forecast_y"])
            )

    def finish(self, duration: float, anomalies: int, possession_resets: int) -> MatchResult:
        self.r.duration = duration
        self.r.anomalies = anomalies
        self.r.possession_resets = possession_resets
        return self.r
