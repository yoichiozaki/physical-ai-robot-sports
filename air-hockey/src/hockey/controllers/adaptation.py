"""Online adaptation of the physical model (system identification during play).

The physics-based planner relies on two table parameters that are unknown on a new
table: the puck's viscous decay rate c (air cushion / friction) and the side-wall
restitution e. ``OnlinePuckEstimator`` identifies both from the observed puck motion:

* free-flight segments (no contact) -> least-squares fit of  ln|v(t)| = a - c t
* side-wall bounces -> e = |v_n after| / |v_n before|  (averaged over a few samples)

Estimates are blended with the prior (nominal model) by evidence weight, so the model
moves smoothly from the prior towards the identified values as data accumulates.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace

import numpy as np

from hockey.controllers.prediction import PuckModel
from hockey.observation import Geometry, PlayerView


@dataclass(frozen=True)
class EstimatorConfig:
    prior_weight_decay: float = 1.0  # seconds of equivalent free-flight evidence
    prior_weight_restitution: float = 2.0  # equivalent number of bounces
    min_speed: float = 0.5
    min_segment: float = 0.12  # s
    max_turn: float = math.radians(8.0)  # direction change that ends a free-flight segment
    wall_band: float = 0.04  # |y| within this of the wall limit counts as a wall bounce


class PuckFilter:
    """Alpha-beta style filter for noisy puck measurements with innovation gating.

    Prediction uses the physical model (decay); a large innovation (contact with a mallet
    or wall) resets the filter to the measurement so that impacts are not smoothed away.
    """

    def __init__(self, model: PuckModel, alpha_pos: float = 0.35, alpha_vel: float = 0.25):
        self.model = model
        self.alpha_pos = alpha_pos
        self.alpha_vel = alpha_vel
        self.gate_vel = 0.35
        self.gate_pos = 0.03
        self.reset()

    def reset(self) -> None:
        self.t: float | None = None
        self.p = np.zeros(2)
        self.v = np.zeros(2)
        self._fresh = 0

    def update(
        self, t: float, p_meas: np.ndarray, v_meas: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        if self.t is None or t - self.t > 0.1 or t <= self.t:
            self.t, self.p, self.v = t, p_meas.astype(float).copy(), v_meas.astype(float).copy()
            self._fresh = 2
            return self.p.copy(), self.v.copy()
        dt = t - self.t
        k = math.exp(-self.model.decay * dt)
        p_pred = self.p + self.v * dt
        v_pred = self.v * k
        if (
            self._fresh > 0
            or float(np.hypot(*(v_meas - v_pred))) > self.gate_vel
            or float(np.hypot(*(p_meas - p_pred))) > self.gate_pos
        ):
            # Impact (or just after one): trust the measurement for a couple of samples, the
            # first sample may fall in the middle of a soft contact.
            if self._fresh == 0:
                self._fresh = 2
            else:
                self._fresh -= 1
            self.p, self.v = p_meas.astype(float).copy(), v_meas.astype(float).copy()
        else:
            self.p = p_pred + self.alpha_pos * (p_meas - p_pred)
            self.v = v_pred + self.alpha_vel * (v_meas - v_pred)
        self.t = t
        return self.p.copy(), self.v.copy()


class OnlinePuckEstimator:
    def __init__(self, geom: Geometry, prior: PuckModel, cfg: EstimatorConfig | None = None):
        self.geom = geom
        self.prior = prior
        self.cfg = cfg or EstimatorConfig()
        self.ylim = geom.half_width - geom.puck_radius
        self.reset()

    def reset(self) -> None:
        self._seg: list[tuple[float, float, float, float, float]] = []  # t, x, y, vx, vy
        self._prev_seg: list[tuple[float, float, float, float, float]] = []
        self.c_sum = 0.0
        self.c_w = 0.0
        self.e_sum = 0.0
        self.e_w = 0.0
        self.n_segments = 0
        self.n_bounces = 0

    @property
    def model(self) -> PuckModel:
        c = self.cfg
        decay = (c.prior_weight_decay * self.prior.decay + self.c_sum) / (
            c.prior_weight_decay + self.c_w
        )
        e = (c.prior_weight_restitution * self.prior.wall_restitution + self.e_sum) / (
            c.prior_weight_restitution + self.e_w
        )
        return replace(
            self.prior,
            decay=float(np.clip(decay, 0.0, 2.0)),
            wall_restitution=float(np.clip(e, 0.3, 1.0)),
        )

    def _close_segment(self) -> None:
        seg = self._seg
        if len(seg) >= 4:
            t = np.array([s[0] for s in seg])
            sp = np.array([math.hypot(s[3], s[4]) for s in seg])
            dur = t[-1] - t[0]
            if dur >= self.cfg.min_segment and sp.min() >= self.cfg.min_speed:
                slope = np.polyfit(t - t[0], np.log(sp), 1)[0]
                w = dur
                self.c_sum += w * float(-slope)
                self.c_w += w
                self.n_segments += 1
        self._prev_seg = seg
        self._seg = []

    def _check_bounce(self) -> None:
        a, b = self._prev_seg, self._seg
        if len(a) < 4 or len(b) < 4:
            return
        # samples away from the contact itself
        vy_a = np.mean([s[4] for s in a[-4:-1]])
        vy_b = np.mean([s[4] for s in b[1:4]])
        vx_a = np.mean([s[3] for s in a[-4:-1]])
        vx_b = np.mean([s[3] for s in b[1:4]])
        y_edge = a[-1][2]
        if abs(abs(y_edge) - self.ylim) > self.cfg.wall_band:
            return
        if vy_a * vy_b >= 0 or abs(vy_a) < 0.3 or abs(vy_b) < 0.1:
            return
        if not (0.85 < (vx_b / vx_a if abs(vx_a) > 0.2 else 1.0) < 1.05):
            return  # tangential velocity changed: not a clean wall bounce (mallet involved)
        e = abs(vy_b) / abs(vy_a)
        if 0.3 < e < 1.05:
            self.e_sum += e
            self.e_w += 1.0
            self.n_bounces += 1

    def update(self, view: PlayerView) -> bool:
        """Feed one observation; returns True when the estimate changed."""
        t = view.time
        x, y = float(view.puck_pos[0]), float(view.puck_pos[1])
        vx, vy = float(view.puck_vel[0]), float(view.puck_vel[1])
        sample = (t, x, y, vx, vy)
        before = (self.n_segments, self.n_bounces)
        if self._seg:
            pt, _, _, pvx, pvy = self._seg[-1]
            sp, psp = math.hypot(vx, vy), math.hypot(pvx, pvy)
            turn = (
                abs(math.atan2(pvx * vy - pvy * vx, pvx * vx + pvy * vy))
                if sp > 1e-6 and psp > 1e-6
                else 0.0
            )
            ratio = sp / psp if psp > 1e-6 else 1.0
            jump = t - pt > 0.05 or math.hypot(x - self._seg[-1][1], y - self._seg[-1][2]) > 0.3
            if turn > self.cfg.max_turn or not (0.85 < ratio < 1.1) or jump:
                self._close_segment()
                if not jump:
                    self._seg = [sample]
                    return False
        self._seg.append(sample)
        if len(self._seg) == 4:
            self._check_bounce()
        return (self.n_segments, self.n_bounces) != before
