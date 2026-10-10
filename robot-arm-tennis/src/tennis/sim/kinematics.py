"""Analytic kinematics of the gantry + 6-DOF arm, frame conversions.

Joint vector (8): q = [gx, gy, q1, q2, q3, q4, q5, q6] in the robot *root frame*.
The arm is an elbow manipulator with a spherical wrist:

    shoulder S = (gx, gy, h0 + d1)
    wrist    W = S + Rz(q1) [a2 (cos q2, 0, sin q2) + a3 (cos(q2+q3), 0, sin(q2+q3))]
    R_tool     = Rz(q1) Ry(-(q2+q3)) Rx(q4) Ry(-q5) Rx(q6)
    racket   P = W + dt * x_tool      (x_tool: handle direction, z_tool: face normal)

``ik`` solves the 5-D task "face centre at P with face normal n" in closed form: the free
roll about n is fixed by pointing the handle away from the shoulder (a natural forehand).
"""

from __future__ import annotations

import math

import numpy as np

from tennis.sim.config import ArmConfig, GantryConfig

TWO_PI = 2.0 * math.pi


def rot_x(a: float) -> np.ndarray:
    c, s = math.cos(a), math.sin(a)
    return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]])


def rot_y(a: float) -> np.ndarray:
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])


def rot_z(a: float) -> np.ndarray:
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def wrap(a: float) -> float:
    return (a + math.pi) % TWO_PI - math.pi


# ----------------------------------------------------------------------------- frames
_ROT_PI = np.array([-1.0, -1.0, 1.0])


def world_to_player(player: str, v: np.ndarray) -> np.ndarray:
    """Positions/velocities: the away player's frame is the world rotated by 180 deg."""
    v = np.asarray(v, dtype=float)
    return v.copy() if player == "home" else v * _ROT_PI


player_to_world = world_to_player  # the rotation is its own inverse


class ArmKinematics:
    def __init__(self, arm: ArmConfig, gantry: GantryConfig) -> None:
        self.h0 = arm.pedestal_height
        self.d1 = arm.shoulder_height
        self.a2, self.a3, self.dt = arm.upper_arm, arm.forearm, arm.tool_length
        self.lo = np.radians(arm.lower_deg)
        self.hi = np.radians(arm.upper_deg)
        self.root_x = gantry.root_x
        self.gx_range, self.gy_range = gantry.x_range, gantry.y_range

    # ------------------------------------------------------------------ frames
    def player_to_root(self, p: np.ndarray) -> np.ndarray:
        return np.asarray(p, dtype=float) + np.array([self.root_x, 0.0, 0.0])

    def root_to_player(self, p: np.ndarray) -> np.ndarray:
        return np.asarray(p, dtype=float) - np.array([self.root_x, 0.0, 0.0])

    def shoulder(self, gx: float, gy: float) -> np.ndarray:
        return np.array([gx, gy, self.h0 + self.d1])

    # ------------------------------------------------------------------ forward
    def fk(self, q: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Returns (racket centre, racket rotation, wrist centre) in the root frame."""
        gx, gy, q1, q2, q3, q4, q5, q6 = (float(v) for v in q)
        r1 = rot_z(q1)
        s = self.shoulder(gx, gy)
        w = s + r1 @ np.array(
            [
                self.a2 * math.cos(q2) + self.a3 * math.cos(q2 + q3),
                0.0,
                self.a2 * math.sin(q2) + self.a3 * math.sin(q2 + q3),
            ]
        )
        r = r1 @ rot_y(-(q2 + q3)) @ rot_x(q4) @ rot_y(-q5) @ rot_x(q6)
        return w + self.dt * r[:, 0], r, w

    def jacobian(self, q: np.ndarray, eps: float = 1e-6) -> np.ndarray:
        """6x6 geometric Jacobian of the racket centre w.r.t. the arm joints (finite diff.)."""
        p0, r0, _ = self.fk(q)
        jac = np.zeros((6, 6))
        for i in range(6):
            dq = np.array(q, dtype=float)
            dq[2 + i] += eps
            p1, r1, _ = self.fk(dq)
            jac[:3, i] = (p1 - p0) / eps
            dr = r1 @ r0.T
            jac[3:, i] = np.array(
                [dr[2, 1] - dr[1, 2], dr[0, 2] - dr[2, 0], dr[1, 0] - dr[0, 1]]
            ) / (2 * eps)
        return jac

    # ------------------------------------------------------------------ inverse
    def handle_direction(
        self, p_site: np.ndarray, n: np.ndarray, gx: float, gy: float
    ) -> np.ndarray | None:
        u = p_site - self.shoulder(gx, gy)
        u[2] *= 0.5  # prefer a fairly horizontal handle
        x_t = u - float(u @ n) * n
        norm = float(np.linalg.norm(x_t))
        if norm < 0.08:
            return None
        return x_t / norm

    def ik(
        self,
        p_site: np.ndarray,
        normal: np.ndarray,
        gantry_xy: tuple[float, float],
        q_ref: np.ndarray | None = None,
        margin: float = 0.03,
    ) -> np.ndarray | None:
        """Closed-form IK (root frame). Returns q (8,) or None if unreachable / out of limits."""
        gx, gy = float(gantry_xy[0]), float(gantry_xy[1])
        n = np.asarray(normal, dtype=float)
        n = n / np.linalg.norm(n)
        x_t = self.handle_direction(np.asarray(p_site, dtype=float), n, gx, gy)
        if x_t is None:
            return None
        y_t = np.cross(n, x_t)
        r_d = np.column_stack([x_t, y_t, n])
        w = p_site - self.dt * x_t
        d = w - self.shoulder(gx, gy)
        q1 = math.atan2(d[1], d[0])
        r = math.hypot(d[0], d[1])
        h = float(d[2])
        c3 = (r * r + h * h - self.a2**2 - self.a3**2) / (2.0 * self.a2 * self.a3)
        if not -0.999 <= c3 <= 0.999:
            return None
        q3 = -math.acos(c3)  # elbow up
        q2 = math.atan2(h, r) - math.atan2(self.a3 * math.sin(q3), self.a2 + self.a3 * math.cos(q3))
        r36 = (rot_z(q1) @ rot_y(-(q2 + q3))).T @ r_d
        sb = math.hypot(r36[0, 1], r36[0, 2])
        b = math.atan2(sb, float(r36[0, 0]))
        if sb < 1e-5:
            return None  # wrist singularity
        a = math.atan2(r36[1, 0], -r36[2, 0])
        c = math.atan2(r36[0, 1], r36[0, 2])
        best = None
        for q4, q5, q6 in ((a, -b, c), (wrap(a + math.pi), b, wrap(c + math.pi))):
            q = np.array([gx, gy, q1, q2, q3, wrap(q4), q5, wrap(q6)])
            if np.any(q[2:] < self.lo + margin) or np.any(q[2:] > self.hi - margin):
                continue
            cost = 0.0 if q_ref is None else float(np.sum(np.abs(q[5:] - np.asarray(q_ref)[5:])))
            if best is None or cost < best[0]:
                best = (cost, q)
        return None if best is None else best[1]

    def gantry_in_range(self, gx: float, gy: float, tol: float = 0.0) -> bool:
        return (
            self.gx_range[0] - tol <= gx <= self.gx_range[1] + tol
            and self.gy_range[0] - tol <= gy <= self.gy_range[1] + tol
        )
