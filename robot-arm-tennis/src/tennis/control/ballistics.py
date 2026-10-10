"""Ball-flight prediction and shot solving (player frame, SI units).

Flight model (between contacts), with relative air velocity u = v - w (wind w):

    a = g - k_d |u| u + k_m (spin x u)

``k_d``/``k_m`` correspond to MuJoCo's ellipsoid fluid model for a sphere
(drag = blunt * rho * pi r^2 |u| u / m, Magnus = C_M * rho * V * spin x u / m) and are
identified against the simulator in ``scripts/calibrate.py``. Spin is not observable by
the robots' sensors, so the physics-based player assumes zero spin.

Court bounce (ball centre reaches z = r): normal velocity -> -e_z * v_z; tangential
velocity loses min(mu (1+e_z) |v_z|, (1-beta) |v_t|) (sliding vs. rolling bounce).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

G = 9.81


@dataclass(frozen=True)
class BallModel:
    radius: float = 0.033
    # Identified against MuJoCo (scripts/calibrate.py -> results/calibration.json)
    k_drag: float = 0.02025  # [1/m]
    k_magnus: float = 0.00148  # [-]
    bounce_e: float = 0.735
    bounce_mu: float = 0.284
    bounce_beta: float = 0.802  # tangential speed kept when the ball grips
    wind: tuple[float, float, float] = (0.0, 0.0, 0.0)
    gravity: float = G


@dataclass(frozen=True)
class CourtGeometry:
    half_length: float = 3.0
    half_width: float = 1.2
    net_height: float = 0.35
    net_half_span: float = 1.45


@dataclass
class FlightEvent:
    kind: str  # "bounce" | "net" | "end"
    t: float
    pos: tuple[float, float, float]
    vel: tuple[float, float, float]


@dataclass
class Prediction:
    """Sampled trajectory (every ``dt``) with events; arrays in the player frame."""

    dt: float
    t: np.ndarray
    pos: np.ndarray  # (N, 3)
    vel: np.ndarray  # (N, 3)
    events: list[FlightEvent] = field(default_factory=list)
    n_bounce_before: np.ndarray | None = None  # bounces that happened before each sample

    def first(self, kind: str) -> FlightEvent | None:
        return next((e for e in self.events if e.kind == kind), None)

    def bounces(self) -> list[FlightEvent]:
        return [e for e in self.events if e.kind == "bounce"]

    def state_at(self, t: float) -> tuple[np.ndarray, np.ndarray]:
        i = min(max(int(t / self.dt), 0), len(self.t) - 2)
        a = (t - self.t[i]) / self.dt
        return self.pos[i] * (1 - a) + self.pos[i + 1] * a, self.vel[i] * (1 - a) + self.vel[
            i + 1
        ] * a


class BallPredictor:
    def __init__(
        self, model: BallModel | None = None, court: CourtGeometry | None = None, dt: float = 0.005
    ):
        self.model = model or BallModel()
        self.court = court or CourtGeometry()
        self.dt = dt

    def set_model(self, model: BallModel) -> None:
        self.model = model

    def _acc(self, vx, vy, vz, sx, sy, sz):
        m = self.model
        ux, uy, uz = vx - m.wind[0], vy - m.wind[1], vz - m.wind[2]
        sp = math.sqrt(ux * ux + uy * uy + uz * uz)
        k, km = m.k_drag * sp, m.k_magnus
        return (
            -k * ux + km * (sy * uz - sz * uy),
            -k * uy + km * (sz * ux - sx * uz),
            -m.gravity - k * uz + km * (sx * uy - sy * ux),
        )

    def predict(
        self,
        pos: np.ndarray,
        vel: np.ndarray,
        horizon: float = 2.0,
        spin: np.ndarray | None = None,
        max_bounces: int = 2,
        stop_at_x: float | None = None,
    ) -> Prediction:
        """Integrate (RK2 midpoint) up to ``horizon``, ``max_bounces`` bounces or the net."""
        m, c, dt = self.model, self.court, self.dt
        r = m.radius
        x, y, z = (float(v) for v in pos)
        vx, vy, vz = (float(v) for v in vel)
        sx, sy, sz = (0.0, 0.0, 0.0) if spin is None else (float(v) for v in spin)
        # hot loop: model terms as locals, acceleration inlined (same arithmetic as _acc)
        w0, w1, w2 = (float(v) for v in m.wind)
        kd, km, g = m.k_drag, m.k_magnus, m.gravity
        half = 0.5 * dt
        n = int(horizon / dt) + 1
        P: list[tuple[float, float, float]] = []
        V: list[tuple[float, float, float]] = []
        nb: list[int] = []
        events: list[FlightEvent] = []
        bounces = 0
        for i in range(n):
            P.append((x, y, z))
            V.append((vx, vy, vz))
            nb.append(bounces)
            if i == n - 1:
                break
            ux, uy, uz = vx - w0, vy - w1, vz - w2
            k = kd * math.sqrt(ux * ux + uy * uy + uz * uz)
            ax = -k * ux + km * (sy * uz - sz * uy)
            ay = -k * uy + km * (sz * ux - sx * uz)
            az = -g - k * uz + km * (sx * uy - sy * ux)
            hx, hy, hz = vx + half * ax, vy + half * ay, vz + half * az
            ux, uy, uz = hx - w0, hy - w1, hz - w2
            k = kd * math.sqrt(ux * ux + uy * uy + uz * uz)
            ax = -k * ux + km * (sy * uz - sz * uy)
            ay = -k * uy + km * (sz * ux - sx * uz)
            az = -g - k * uz + km * (sx * uy - sy * ux)
            nx_, ny_, nz_ = x + dt * hx, y + dt * hy, z + dt * hz
            nvx, nvy, nvz = vx + dt * ax, vy + dt * ay, vz + dt * az
            # net (plane x = 0)
            if (x < 0.0) != (nx_ < 0.0) and abs(x - nx_) > 1e-12:
                f = x / (x - nx_)
                zc, yc = z + f * (nz_ - z), y + f * (ny_ - y)
                if abs(yc) <= c.net_half_span and zc < c.net_height + r:
                    events.append(FlightEvent("net", (i + f) * dt, (0.0, yc, zc), (vx, vy, vz)))
                    break
            if nz_ <= r and nvz < 0.0:
                f = (z - r) / max(z - nz_, 1e-12)
                tb = (i + f) * dt
                bx, by = x + f * (nx_ - x), y + f * (ny_ - y)
                bvx, bvy, bvz = vx + f * (nvx - vx), vy + f * (nvy - vy), vz + f * (nvz - vz)
                events.append(FlightEvent("bounce", tb, (bx, by, r), (bvx, bvy, bvz)))
                bounces += 1
                vt = math.hypot(bvx, bvy)
                vz_out = -m.bounce_e * bvz
                if vt > 1e-9:
                    loss = min(
                        m.bounce_mu * (1.0 + m.bounce_e) * abs(bvz), (1.0 - m.bounce_beta) * vt
                    )
                    scale = (vt - loss) / vt
                else:
                    scale = 1.0
                # continue the remaining (1 - f) of the step after the bounce
                rem = (1.0 - f) * dt
                nvx, nvy, nvz = bvx * scale, bvy * scale, vz_out
                nx_, ny_, nz_ = bx + rem * nvx, by + rem * nvy, r + rem * nvz
                if bounces >= max_bounces:
                    P.append((nx_, ny_, nz_))
                    V.append((nvx, nvy, nvz))
                    nb.append(bounces)
                    break
            if stop_at_x is not None and (nx_ - stop_at_x) * (x - stop_at_x) < 0.0:
                P.append((nx_, ny_, nz_))
                V.append((nvx, nvy, nvz))
                nb.append(bounces)
                break
            x, y, z, vx, vy, vz = nx_, ny_, nz_, nvx, nvy, nvz
        i_end = min(len(P), n)
        Pa = np.array(P[:i_end])
        Va = np.array(V[:i_end])
        events.append(FlightEvent("end", (i_end - 1) * dt, tuple(Pa[-1]), tuple(Va[-1])))
        return Prediction(
            dt, np.arange(i_end) * dt, Pa, Va, events, np.array(nb[:i_end], dtype=np.int8)
        )

    # ------------------------------------------------------------------ shots
    def landing(
        self, pos: np.ndarray, vel: np.ndarray, spin: np.ndarray | None = None
    ) -> tuple[FlightEvent | None, float | None]:
        """First bounce of a shot and the ball-centre height when crossing the net plane."""
        pred = self.predict(pos, vel, horizon=3.0, spin=spin, max_bounces=1)
        net_z = None
        xs = pred.pos[:, 0]
        cross = np.nonzero((xs[:-1] < 0.0) & (xs[1:] >= 0.0))[0]
        if len(cross):
            i = int(cross[0])
            f = -xs[i] / (xs[i + 1] - xs[i])
            net_z = float(pred.pos[i, 2] + f * (pred.pos[i + 1, 2] - pred.pos[i, 2]))
        if pred.first("net") is not None:
            return None, net_z
        return pred.first("bounce"), net_z

    def solve_shot(
        self,
        p0: np.ndarray,
        target: tuple[float, float],
        speed: float,
        net_margin: float = 0.10,
        spin: np.ndarray | None = None,
        iters: int = 4,
    ) -> tuple[np.ndarray, dict] | None:
        """Outgoing velocity that makes the ball's first bounce land at ``target`` (x, y).

        ``speed`` is the horizontal speed of the drag-free guess; the arc is raised until the
        ball clears the net by ``net_margin``. Drag (and spin, if given) are compensated by
        iterating on the aim point.
        """
        r = self.model.radius
        x0, y0, z0 = (float(v) for v in p0)
        tx, ty = target
        for _arc in range(6):
            aim = np.array([tx, ty], dtype=float)
            best = None
            for _ in range(iters):
                dx, dy = aim[0] - x0, aim[1] - y0
                dist = math.hypot(dx, dy)
                T = max(dist / max(speed, 0.5), 0.25)
                v0 = np.array([dx / T, dy / T, (r - z0) / T + 0.5 * self.model.gravity * T])
                bounce, net_z = self.landing(p0, v0, spin)
                if bounce is None:
                    break
                err = np.array([bounce.pos[0] - tx, bounce.pos[1] - ty])
                best = (v0, bounce, net_z, err)
                if float(np.hypot(*err)) < 0.01:
                    break
                aim = aim - err
            if best is not None:
                v0, bounce, net_z, err = best
                if (
                    net_z is not None
                    and net_z >= self.court.net_height + r + net_margin
                    and float(np.hypot(*err)) < 0.08
                ):
                    return v0, {
                        "landing": bounce.pos,
                        "net_z": net_z,
                        "flight_time": bounce.t,
                        "error": float(np.hypot(*err)),
                    }
            speed *= 0.85  # raise the arc: slower horizontal speed, longer flight
        return None
