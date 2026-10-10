"""Low-level motion control of the 2-DOF gantry robots.

Every high-level controller (reactive, physics-based, RL, hybrid) outputs a *target
position* for its mallet. ``MotionController`` turns that into a velocity command for
the MuJoCo velocity servos at the physics rate, enforcing per-axis position, velocity
and acceleration limits (a time-optimal "brake-before-target" profile followed by a
slew-rate limiter). The servos themselves are force-limited in the MJCF, so the robot
can never exceed its physical capabilities even under contact.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

SOFT_MARGIN = 0.004  # software limits sit 4 mm inside the hard joint limits


@dataclass(frozen=True)
class AxisLimits:
    qmin: float
    qmax: float
    vmax: float
    amax: float

    def scaled(self, speed_scale: float) -> AxisLimits:
        return AxisLimits(self.qmin, self.qmax, self.vmax * speed_scale, self.amax * speed_scale)


class MotionController:
    """Per-axis target-position tracker with velocity/acceleration/position limits."""

    def __init__(
        self,
        x: AxisLimits,
        y: AxisLimits,
        position_gain: float = 40.0,
        brake_ratio: float = 0.85,
    ) -> None:
        self.base_limits = (x, y)
        self.limits = (x, y)
        self.kp = position_gain
        self.brake_ratio = brake_ratio
        self.u = [0.0, 0.0]
        self.clipped_last = False  # target was outside the workspace on the last call

    def set_speed_scale(self, scale: float) -> None:
        self.limits = tuple(lim.scaled(scale) for lim in self.base_limits)  # type: ignore[assignment]

    def reset(self) -> None:
        self.u = [0.0, 0.0]
        self.clipped_last = False

    def clip_target(self, tx: float, ty: float) -> tuple[float, float]:
        """Clip to the soft workspace (``SOFT_MARGIN`` inside the hard joint limits)."""
        lx, ly = self.limits
        m = SOFT_MARGIN
        return min(max(tx, lx.qmin + m), lx.qmax - m), min(max(ty, ly.qmin + m), ly.qmax - m)

    def command(self, tx: float, ty: float, qx: float, qy: float, dt: float) -> tuple[float, float]:
        """Velocity command towards (tx, ty).

        Coordinated (straight-line) motion: a 1-D brake-before-target profile along the
        line to the target, with path limits derived from the per-axis limits, followed by
        a per-axis slew-rate (acceleration) limiter. Straight-line motion makes the mallet
        velocity at impact point along the planned strike direction.
        """
        cx, cy = self.clip_target(tx, ty)
        self.clipped_last = abs(cx - tx) > 1e-9 or abs(cy - ty) > 1e-9
        lx, ly = self.limits
        ex, ey = cx - qx, cy - qy
        dist = math.hypot(ex, ey)
        if dist > 1e-9:
            ux, uy = ex / dist, ey / dist
            ax_, ay_ = abs(ux), abs(uy)
            vline = min(
                lx.vmax / ax_ if ax_ > 1e-9 else math.inf, ly.vmax / ay_ if ay_ > 1e-9 else math.inf
            )
            aline = min(
                lx.amax / ax_ if ax_ > 1e-9 else math.inf, ly.amax / ay_ if ay_ > 1e-9 else math.inf
            )
            speed = min(vline, math.sqrt(2.0 * self.brake_ratio * aline * dist), self.kp * dist)
            vx, vy = speed * ux, speed * uy
        else:
            vx = vy = 0.0
        # Coordinated slew-rate limit: scale the velocity change uniformly so that every axis
        # respects its acceleration limit and the acceleration keeps its direction (straight
        # strike lines even though the axes have different limits).
        px, py = self.u
        dx, dy = vx - px, vy - py
        k = 1.0
        if abs(dx) > 1e-12:
            k = min(k, lx.amax * dt / abs(dx))
        if abs(dy) > 1e-12:
            k = min(k, ly.amax * dt / abs(dy))
        nx_ = min(max(px + k * dx, -lx.vmax), lx.vmax)
        ny_ = min(max(py + k * dy, -ly.vmax), ly.vmax)
        self.u = [nx_, ny_]
        return nx_, ny_


def min_time_1d(d: float, v0: float, vmax: float, amax: float) -> float:
    """Minimum time to travel signed distance ``d`` from velocity ``v0`` (final velocity free).

    Bang-bang with velocity saturation: brake if moving away, then accelerate to vmax.
    """
    if d < 0.0:
        d, v0 = -d, -v0
    t = 0.0
    if v0 < 0.0:  # moving away: stop first
        t_stop = -v0 / amax
        d += v0 * v0 / (2.0 * amax)
        t += t_stop
        v0 = 0.0
    v0 = min(v0, vmax)
    t_acc = (vmax - v0) / amax
    s_acc = (vmax * vmax - v0 * v0) / (2.0 * amax)
    if s_acc >= d:
        return t + (-v0 + math.sqrt(v0 * v0 + 2.0 * amax * d)) / amax
    return t + t_acc + (d - s_acc) / vmax


def min_time_to_reach(
    pos: tuple[float, float],
    vel: tuple[float, float],
    target: tuple[float, float],
    limits: tuple[AxisLimits, AxisLimits],
) -> float:
    """Gantry axes are independent: the reach time is the slower axis."""
    tx = min_time_1d(target[0] - pos[0], vel[0], limits[0].vmax, limits[0].amax)
    ty = min_time_1d(target[1] - pos[1], vel[1], limits[1].vmax, limits[1].amax)
    return max(tx, ty)


def rest_time_1d(d: float, v0: float, vmax: float, amax: float, brake: float) -> float:
    """Minimum time to travel ``d`` from velocity ``v0`` and *stop* there.

    Accelerate at ``amax`` (cruise at ``vmax``), brake at ``brake`` (<= amax).
    """
    if d < 0.0:
        d, v0 = -d, -v0
    t = 0.0
    if v0 < 0.0:  # moving away: stop first
        t += -v0 / amax
        d += v0 * v0 / (2.0 * amax)
        v0 = 0.0
    stop = v0 * v0 / (2.0 * brake)
    if stop > d:  # cannot stop in time: brake, then come back
        t += v0 / brake
        back = stop - d
        vp = math.sqrt(back / (1.0 / (2.0 * amax) + 1.0 / (2.0 * brake)))
        vp = min(vp, vmax)
        return (
            t
            + vp / amax
            + vp / brake
            + max(0.0, back - vp * vp / (2 * amax) - vp * vp / (2 * brake)) / vmax
        )
    vp = math.sqrt((d + v0 * v0 / (2.0 * amax)) / (1.0 / (2.0 * amax) + 1.0 / (2.0 * brake)))
    if vp <= vmax:
        return t + (vp - v0) / amax + vp / brake
    s_acc = (vmax * vmax - v0 * v0) / (2.0 * amax)
    s_dec = vmax * vmax / (2.0 * brake)
    return t + (vmax - v0) / amax + (d - s_acc - s_dec) / vmax + vmax / brake


def time_to_stop_at(
    pos: tuple[float, float],
    vel: tuple[float, float],
    target: tuple[float, float],
    limits: tuple[AxisLimits, AxisLimits],
    brake_ratio: float = 0.85,
) -> float:
    """Rest-to-rest reach time (slower axis) used to time the arrival at a staging point."""
    lx, ly = limits
    tx = rest_time_1d(target[0] - pos[0], vel[0], lx.vmax, lx.amax, brake_ratio * lx.amax)
    ty = rest_time_1d(target[1] - pos[1], vel[1], ly.vmax, ly.amax, brake_ratio * ly.amax)
    return max(tx, ty)
