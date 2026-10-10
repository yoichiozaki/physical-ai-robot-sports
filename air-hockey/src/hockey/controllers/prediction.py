"""Analytic puck trajectory prediction with viscous decay and wall reflections.

Model (between impacts):  dv/dt = -c v   =>  v(t) = v0 e^{-ct},  p(t) = p0 + v0 (1 - e^{-ct}) / c
Walls: the puck centre is confined to |x| <= L/2 - r, |y| <= W/2 - r; an impact flips the
normal velocity component and scales it by the restitution e (frictionless contacts keep
the tangential component). Crossing an end wall inside the goal mouth ends the trajectory
as a goal. Mallets are not modelled (they are what the planner decides about).

All arithmetic uses Python floats: the predictor runs every control step (50 Hz) for
several players and must be cheap.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from hockey.observation import Geometry

STOP_SPEED = 0.02
FORECAST_HORIZON = 0.5  # [s] horizon of the forecast used for the prediction-error metric


@dataclass(frozen=True)
class PuckModel:
    decay: float = 0.15
    wall_restitution: float = 0.85
    mallet_restitution: float = 0.68  # identified from the simulator (servo-held mallet)


@dataclass
class Segment:
    t0: float
    x0: float
    y0: float
    vx0: float
    vy0: float
    t1: float
    end: str  # "wall_side" | "wall_end" | "goal_own" | "goal_opp" | "stop" | "horizon"


@dataclass
class Trajectory:
    segments: list[Segment] = field(default_factory=list)
    decay: float = 0.0

    def _free(self, seg: Segment, t: float) -> tuple[float, float, float, float]:
        tau = t - seg.t0
        c = self.decay
        if c > 1e-9:
            k = math.exp(-c * tau)
            f = (1.0 - k) / c
        else:
            k, f = 1.0, tau
        return seg.x0 + seg.vx0 * f, seg.y0 + seg.vy0 * f, seg.vx0 * k, seg.vy0 * k

    def state_at(self, t: float) -> tuple[float, float, float, float]:
        for seg in self.segments:
            if t <= seg.t1:
                return self._free(seg, max(t, seg.t0))
        last = self.segments[-1]
        return self._free(last, last.t1)

    @property
    def end_time(self) -> float:
        return self.segments[-1].t1 if self.segments else 0.0

    @property
    def end_event(self) -> str:
        return self.segments[-1].end if self.segments else "stop"

    def bounce_points(self) -> list[tuple[float, float]]:
        pts = []
        for seg in self.segments[:-1]:
            x, y, _, _ = self._free(seg, seg.t1)
            pts.append((x, y))
        return pts

    def sample(self, dt: float, t_max: float | None = None) -> np.ndarray:
        """Array (N, 5): t, x, y, vx, vy sampled every ``dt`` up to the end of the trajectory."""
        t_end = self.end_time if t_max is None else min(t_max, self.end_time)
        n = max(1, int(t_end / dt) + 1)
        t = np.arange(n) * dt
        out = np.empty((n, 5))
        out[:, 0] = t
        c = self.decay
        start = 0
        for k, seg in enumerate(self.segments):
            last = k == len(self.segments) - 1
            stop = n if last else int(np.searchsorted(t, seg.t1, side="right"))
            if stop <= start:
                continue
            tau = np.minimum(t[start:stop], seg.t1) - seg.t0
            if c > 1e-9:
                kk = np.exp(-c * tau)
                f = (1.0 - kk) / c
            else:
                kk, f = np.ones_like(tau), tau
            out[start:stop, 1] = seg.x0 + seg.vx0 * f
            out[start:stop, 2] = seg.y0 + seg.vy0 * f
            out[start:stop, 3] = seg.vx0 * kk
            out[start:stop, 4] = seg.vy0 * kk
            start = stop
            if start >= n:
                break
        return out

    def first_crossing_x(self, x_line: float) -> tuple[float, float, float, float] | None:
        """First time the puck centre crosses ``x_line`` while moving towards -x.

        Returns (t, y, vx, vy) or None.
        """
        c = self.decay
        for seg in self.segments:
            if seg.vx0 >= 0.0 or seg.x0 < x_line:
                continue
            dist = x_line - seg.x0  # negative
            tau = _time_to_disp(seg.vx0, dist, c)
            if tau is None or seg.t0 + tau > seg.t1 + 1e-9:
                continue
            _x, y, vx, vy = self._free(seg, seg.t0 + tau)
            return seg.t0 + tau, y, vx, vy
        return None


def _time_to_disp(v0: float, disp: float, c: float) -> float | None:
    """Time for the exponentially decaying motion with initial velocity v0 to cover ``disp``."""
    if v0 == 0.0 or disp == 0.0 or (v0 > 0) != (disp > 0):
        return None if disp != 0.0 else 0.0
    if c <= 1e-9:
        return disp / v0
    ratio = c * disp / v0
    if ratio >= 1.0:
        return None
    return -math.log1p(-ratio) / c


class PuckPredictor:
    def __init__(
        self, geom: Geometry, model: PuckModel | None = None, max_bounces: int = 6
    ) -> None:
        self.geom = geom
        self.model = model or PuckModel()
        self.max_bounces = max_bounces
        self.x_lim = geom.half_length - geom.puck_radius
        self.y_lim = geom.half_width - geom.puck_radius
        self.goal_half = geom.goal_width / 2

    def set_model(self, model: PuckModel) -> None:
        self.model = model

    def predict(self, pos: np.ndarray, vel: np.ndarray, horizon: float = 2.0) -> Trajectory:
        c = self.model.decay
        e = self.model.wall_restitution
        xl, yl = self.x_lim, self.y_lim
        x, y = float(pos[0]), float(pos[1])
        vx, vy = float(vel[0]), float(vel[1])
        traj = Trajectory(decay=c)
        t = 0.0
        for _ in range(self.max_bounces + 1):
            # Already in contact with a wall and moving into it: reflect first.
            if (y >= yl and vy > 0) or (y <= -yl and vy < 0):
                vy = -e * vy
            if ((x >= xl and vx > 0) or (x <= -xl and vx < 0)) and abs(y) >= self.goal_half:
                vx = -e * vx
            if math.hypot(vx, vy) < STOP_SPEED:
                traj.segments.append(Segment(t, x, y, 0.0, 0.0, horizon, "stop"))
                return traj
            t_y = _time_to_disp(vy, (yl if vy > 0 else -yl) - y, c) if vy != 0 else None
            t_x = _time_to_disp(vx, (xl if vx > 0 else -xl) - x, c) if vx != 0 else None
            t_y = math.inf if t_y is None else max(t_y, 0.0)
            t_x = math.inf if t_x is None else max(t_x, 0.0)
            remaining = horizon - t
            dt_evt = min(t_x, t_y)
            if dt_evt >= remaining:
                seg = Segment(t, x, y, vx, vy, horizon, "horizon")
                # A decaying puck that never reaches a wall "stops" (asymptotically).
                if math.isinf(dt_evt):
                    seg.end = "horizon"
                traj.segments.append(seg)
                return traj
            seg = Segment(t, x, y, vx, vy, t + dt_evt, "")
            x, y, vx, vy = traj._free(seg, t + dt_evt)
            t += dt_evt
            if t_x <= t_y:
                if abs(y) < self.goal_half:
                    seg.end = "goal_own" if vx < 0 else "goal_opp"
                    traj.segments.append(seg)
                    return traj
                seg.end = "wall_end"
                x = xl if vx > 0 else -xl
                vx = -e * vx
            else:
                seg.end = "wall_side"
                y = yl if vy > 0 else -yl
                vy = -e * vy
            traj.segments.append(seg)
        last = traj.segments[-1]
        traj.segments.append(Segment(last.t1, x, y, vx, vy, max(horizon, last.t1), "horizon"))
        return traj
