"""Interception, shot and defence planning in the player frame.

Key physics used by the attack planner (frictionless impact, mallet much heavier than
the puck, restitution e_m):

    v_out = v_in + (1 + e_m) * ((v_m - v_in) . n) n

To send the puck along unit direction d with speed mu, the contact normal must be
n = normalize(mu d - v_in) and the mallet speed along n must be

    s_n = v_in . n + |mu d - v_in| / (1 + e_m).

The mallet centre at impact is c = p - (r_puck + r_mallet) n. The robot first moves to a
staging point behind c (run-up) and then strikes along n with straight-line motion.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import IntEnum

import numpy as np

from hockey.controllers.prediction import Trajectory
from hockey.observation import Geometry
from hockey.simulation.robot import AxisLimits, time_to_stop_at


class Strategy(IntEnum):
    DEFEND = 0
    DIRECT = 1
    BANK_LEFT = 2  # off the +y side wall
    BANK_RIGHT = 3  # off the -y side wall


STRATEGY_LABELS = {
    Strategy.DEFEND: "Defense",
    Strategy.DIRECT: "Direct shot",
    Strategy.BANK_LEFT: "Bank shot (L)",
    Strategy.BANK_RIGHT: "Bank shot (R)",
}
STRATEGY_LABELS_JA = {
    Strategy.DEFEND: "防御",
    Strategy.DIRECT: "ストレート",
    Strategy.BANK_LEFT: "バンク(左壁)",
    Strategy.BANK_RIGHT: "バンク(右壁)",
}


@dataclass(frozen=True)
class ShotSpec:
    strategy: Strategy = Strategy.DIRECT
    aim_y: float = 0.0  # target y on the opponent goal line
    speed: float = 3.5  # desired outgoing puck speed [m/s]


@dataclass
class StrikePlan:
    spec: ShotSpec
    t_contact: float  # seconds from now
    puck_contact: tuple[float, float]
    puck_vel: tuple[float, float]
    direction: tuple[float, float]  # desired outgoing direction of the puck
    normal: tuple[float, float]  # contact normal (mallet -> puck)
    mallet_contact: tuple[float, float]
    mallet_speed: float  # mallet speed along the normal at impact
    staging: tuple[float, float]
    strike_target: tuple[float, float]
    t_strike: float  # duration of the run-up from the staging point
    t_reach: float  # time to reach the staging point from the current state
    slack: float  # t_contact - (t_reach + t_strike + latency)
    expected_out_speed: float


def _norm(x: float, y: float) -> tuple[float, float, float]:
    n = math.hypot(x, y)
    if n < 1e-12:
        return 0.0, 0.0, 0.0
    return x / n, y / n, n


def shot_direction(
    px: float, py: float, spec: ShotSpec, geom: Geometry, wall_restitution: float
) -> tuple[float, float] | None:
    """Unit direction the puck must leave in to reach the aim point (bank shots included).

    For a bank shot with restitution e the slope after the bounce is -e times the slope
    before it, which gives the closed form used below (exact for isotropic viscous decay,
    which does not change the path geometry).
    """
    gx = geom.half_length
    dx = gx - px
    if dx < 0.15:
        return None
    ylim = geom.half_width - geom.puck_radius
    gy = spec.aim_y
    if spec.strategy == Strategy.DIRECT or spec.strategy == Strategy.DEFEND:
        ux, uy, _ = _norm(dx, gy - py)
        return ux, uy
    e = max(wall_restitution, 0.2)
    if spec.strategy == Strategy.BANK_LEFT:
        slope = ((ylim - py) + (ylim - gy) / e) / dx
    else:
        slope = -((ylim + py) + (ylim + gy) / e) / dx
    ux, uy, _ = _norm(1.0, slope)
    return ux, uy


def shot_path(
    px: float, py: float, spec: ShotSpec, geom: Geometry, wall_restitution: float
) -> list[tuple[float, float]]:
    """Polyline of the intended puck path (start, [bounce], goal) for clearance checks/plots."""
    d = shot_direction(px, py, spec, geom, wall_restitution)
    gx = geom.half_length
    if d is None:
        return [(px, py)]
    if spec.strategy in (Strategy.DIRECT, Strategy.DEFEND):
        return [(px, py), (gx, spec.aim_y)]
    ylim = geom.half_width - geom.puck_radius
    yb = ylim if spec.strategy == Strategy.BANK_LEFT else -ylim
    xb = px + (yb - py) * d[0] / d[1] if abs(d[1]) > 1e-9 else gx
    return [(px, py), (xb, yb), (gx, spec.aim_y)]


def _segment_distance(
    px: float, py: float, a: tuple[float, float], b: tuple[float, float]
) -> float:
    ax, ay = a
    bx, by = b
    vx, vy = bx - ax, by - ay
    l2 = vx * vx + vy * vy
    t = 0.0 if l2 < 1e-12 else max(0.0, min(1.0, ((px - ax) * vx + (py - ay) * vy) / l2))
    return math.hypot(px - (ax + t * vx), py - (ay + t * vy))


def path_clearance(path: list[tuple[float, float]], ox: float, oy: float) -> float:
    """Minimum distance between the opponent mallet centre and the intended puck path."""
    if len(path) < 2:
        return 0.0
    return min(_segment_distance(ox, oy, path[i], path[i + 1]) for i in range(len(path) - 1))


def _ray_box_exit(
    x: float, y: float, ux: float, uy: float, xr: tuple[float, float], yr: tuple[float, float]
) -> float:
    """Largest t >= 0 such that (x, y) + t (ux, uy) stays inside the box (0 if outside)."""
    if not (xr[0] - 1e-9 <= x <= xr[1] + 1e-9 and yr[0] - 1e-9 <= y <= yr[1] + 1e-9):
        return 0.0
    t = math.inf
    if ux > 1e-12:
        t = min(t, (xr[1] - x) / ux)
    elif ux < -1e-12:
        t = min(t, (xr[0] - x) / ux)
    if uy > 1e-12:
        t = min(t, (yr[1] - y) / uy)
    elif uy < -1e-12:
        t = min(t, (yr[0] - y) / uy)
    return max(0.0, t)


@dataclass(frozen=True)
class PlannerConfig:
    mallet_restitution: float = 0.68
    wall_restitution: float = 0.85
    latency: float = 0.0  # decision-to-actuation delay (the launch itself is time-stamped)
    reach_margin: float = 0.03  # safety margin on the (optimistic) time to reach the staging point
    servo_lag: float = 0.010  # velocity-servo lag added to the run-up time
    max_run: float = 0.28  # longest run-up distance [m]
    min_run: float = 0.02
    min_mallet_speed: float = 0.3
    max_shot_speed: float = 7.0
    strike_zone_x_max: float = -0.05  # puck centre must be behind this line at impact
    workspace_margin: float = 0.01
    brake_ratio: float = 0.85
    sample_dt: float = 0.02


class StrikePlanner:
    """Finds the earliest feasible strike on a predicted puck trajectory."""

    def __init__(self, geom: Geometry, cfg: PlannerConfig | None = None) -> None:
        self.geom = geom
        self.cfg = cfg or PlannerConfig()
        m = self.cfg.workspace_margin
        self.xr = (geom.x_range[0] + m, geom.x_range[1] - m)
        self.yr = (geom.y_range[0] + m, geom.y_range[1] - m)
        self.r_sum = geom.puck_radius + geom.mallet_radius

    def plan_at(
        self,
        t_contact: float,
        p: tuple[float, float],
        v: tuple[float, float],
        spec: ShotSpec,
        own_pos: np.ndarray,
        own_vel: np.ndarray,
        limits: tuple[AxisLimits, AxisLimits],
    ) -> StrikePlan | None:
        cfg = self.cfg
        px, py = p
        vx, vy = v
        if px > cfg.strike_zone_x_max:
            return None
        d = shot_direction(px, py, spec, self.geom, cfg.wall_restitution)
        if d is None:
            return None
        e_m = cfg.mallet_restitution
        lx, ly = limits
        # Solve for the outgoing speed mu whose required mallet speed is achievable with the
        # available run-up / follow-through room (bisection; the geometry depends on mu).
        mu, mu_lo, mu_hi = spec.speed, 0.2, cfg.max_shot_speed
        for _ in range(10):
            nx, ny, lam = _norm(mu * d[0] - vx, mu * d[1] - vy)
            if lam < 1e-9:
                return None
            s_n = vx * nx + vy * ny + lam / (1.0 + e_m)
            ax_, ay_ = abs(nx), abs(ny)
            vline = min(
                lx.vmax / ax_ if ax_ > 1e-9 else math.inf, ly.vmax / ay_ if ay_ > 1e-9 else math.inf
            )
            aline = min(
                lx.amax / ax_ if ax_ > 1e-9 else math.inf, ly.amax / ay_ if ay_ > 1e-9 else math.inf
            )
            cx, cy = px - self.r_sum * nx, py - self.r_sum * ny
            if not (self.xr[0] <= cx <= self.xr[1] and self.yr[0] <= cy <= self.yr[1]):
                return None
            run_room = _ray_box_exit(cx, cy, -nx, -ny, self.xr, self.yr)
            follow_room = _ray_box_exit(cx, cy, nx, ny, self.xr, self.yr)
            run_max = min(cfg.max_run, run_room)
            s_hi = min(
                0.95 * vline,
                math.sqrt(2.0 * aline * run_max),
                math.sqrt(2.0 * cfg.brake_ratio * aline * max(follow_room, 1e-6)),
            )
            s_lo = min(max(cfg.min_mallet_speed, math.sqrt(2.0 * aline * cfg.min_run)), s_hi)
            if s_n > s_hi + 1e-3:
                mu_hi = mu
                mu = 0.5 * (mu_lo + mu)
            elif s_n < s_lo - 1e-3:
                mu_lo = mu
                mu = 0.5 * (mu + mu_hi)
            else:
                break
        s_n = min(max(s_n, s_lo), s_hi)
        if s_hi < cfg.min_mallet_speed * 0.5:
            return None
        run = min(max(s_n * s_n / (2.0 * aline), cfg.min_run), run_max)
        if run < 0.015:
            return None
        sx, sy = cx - nx * run, cy - ny * run
        follow = min(s_n * s_n / (2.0 * cfg.brake_ratio * aline) + 0.03, follow_room)
        tx, ty = cx + nx * follow, cy + ny * follow
        # Run-up from rest along the line (+ servo lag).
        if s_n < 0.999 * vline:
            t_strike = math.sqrt(2.0 * run / aline)
        else:
            t_strike = run / vline + vline / (2.0 * aline)
        t_strike += cfg.servo_lag
        t_reach = time_to_stop_at(
            (float(own_pos[0]), float(own_pos[1])),
            (float(own_vel[0]), float(own_vel[1])),
            (sx, sy),
            limits,
            cfg.brake_ratio,
        )
        # The reach-time estimate is optimistic (bang-bang); add a margin that vanishes once
        # the mallet waits at the staging point.
        margin = min(t_reach, cfg.reach_margin)
        slack = t_contact - (t_reach + margin + t_strike + cfg.latency)
        # Expected outgoing velocity with the achievable mallet speed.
        k = (1.0 + e_m) * (s_n - (vx * nx + vy * ny))
        out_x, out_y = vx + k * nx, vy + k * ny
        return StrikePlan(
            spec=spec,
            t_contact=t_contact,
            puck_contact=(px, py),
            puck_vel=(vx, vy),
            direction=d,
            normal=(nx, ny),
            mallet_contact=(cx, cy),
            mallet_speed=s_n,
            staging=(sx, sy),
            strike_target=(tx, ty),
            t_strike=t_strike,
            t_reach=t_reach,
            slack=slack,
            expected_out_speed=math.hypot(out_x, out_y),
        )

    def find_strike(
        self,
        traj: Trajectory,
        spec: ShotSpec,
        own_pos: np.ndarray,
        own_vel: np.ndarray,
        limits: tuple[AxisLimits, AxisLimits],
        horizon: float = 1.5,
        min_slack: float = 0.0,
        samples: np.ndarray | None = None,
    ) -> StrikePlan | None:
        """Earliest feasible strike (slack >= ``min_slack``) along the trajectory."""
        if samples is None:
            samples = traj.sample(self.cfg.sample_dt, horizon)
        x_max = self.cfg.strike_zone_x_max
        for t, x, y, vx, vy in samples.tolist():
            if x > x_max:
                continue
            plan = self.plan_at(t, (x, y), (vx, vy), spec, own_pos, own_vel, limits)
            if plan is not None and plan.slack >= min_slack:
                return plan
        return None


def defense_point(traj: Trajectory, geom: Geometry, defense_x: float) -> tuple[float, float] | None:
    """Where (and when) the predicted puck path crosses the defence line, aimed at our goal side."""
    hit = traj.first_crossing_x(defense_x)
    if hit is None:
        return None
    t, y, _vx, _vy = hit
    return t, y


def detour_target(
    own: tuple[float, float],
    goal_pt: tuple[float, float],
    puck: tuple[float, float],
    normal: tuple[float, float],
    clearance: float,
) -> tuple[float, float] | None:
    """Side-step waypoint if the straight path to ``goal_pt`` would touch the puck early."""
    if _segment_distance(puck[0], puck[1], own, goal_pt) >= clearance:
        return None
    nx, ny = normal
    px_, py_ = -ny, nx  # perpendicular to the strike direction
    side = 1.0 if (own[0] - puck[0]) * px_ + (own[1] - puck[1]) * py_ >= 0 else -1.0
    return (
        puck[0] + side * px_ * (clearance + 0.04) - nx * 0.04,
        puck[1] + side * py_ * (clearance + 0.04) - ny * 0.04,
    )
