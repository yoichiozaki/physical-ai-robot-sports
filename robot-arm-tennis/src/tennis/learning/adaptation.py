"""Online adaptation of the physics models from the robot's own observations.

* ``ModelAdapter`` - online system identification of the ball model during play: court
  restitution / friction from every observed bounce (velocity just before and after), and air
  drag + horizontal wind from the observed flight (recursive least squares on the measured
  deceleration). The player's predictor and shot solver are updated in place, so both the
  return (where will the ball be?) and the aim (how must I hit it?) follow the new court.
* ``ImpactAdapter`` - shrinkage estimate of the racket impact gain: the ratio of measured to
  planned outgoing speed along the racket normal rescales the impact model (a, b) -> k (a, b).
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING

import numpy as np

from tennis.control.ballistics import BallModel, G
from tennis.control.planner import HitPlan, SwingPlanner

if TYPE_CHECKING:
    from tennis.control.players import PhysicsPlayer, PlayerView


@dataclass
class ImpactAdapter:
    prior_weight: float = 4.0  # shots worth of confidence in the nominal impact model
    forget: float = 0.9
    k_min: float = 0.75
    k_max: float = 1.35
    deadband: float = 0.06  # |k - 1| below this is treated as "home racket" (noise level)
    min_obs: int = 3
    k: float = 1.0
    n: int = 0
    _sum: float = 0.0
    _w: float = 0.0
    _base: tuple[float, float] | None = None

    @property
    def shifted(self) -> bool:
        return self.n >= self.min_obs and abs(self.k - 1.0) > self.deadband

    def reset(self, planner: SwingPlanner) -> None:
        if self._base is None:
            self._base = (planner.pc.impact_a, planner.pc.impact_b)
        self.k, self.n, self._sum, self._w = 1.0, 0, 0.0, 0.0
        self.apply(planner)

    def apply(self, planner: SwingPlanner) -> None:
        a, b = self._base
        k = self.k if self.shifted else 1.0
        planner.pc = replace(planner.pc, impact_a=a * k, impact_b=b * k)

    def observe(self, plan: HitPlan, v_measured: np.ndarray, dt_since_contact: float, k_drag: float,
                planner: SwingPlanner, gravity: float = G) -> None:  # fmt: skip
        if plan.v_out is None:
            return
        # undo gravity and drag acting since the ball left the racket
        v = np.asarray(v_measured, float)
        acc = np.array([0.0, 0.0, gravity]) + k_drag * float(np.linalg.norm(v)) * v
        v = v + dt_since_contact * acc
        n = plan.normal
        planned_n = float(plan.v_out @ n)
        if planned_n < 1.0:
            return
        # the plan was made with the applied gain -> the observation estimates k_applied * ratio
        k_applied = self.k if self.shifted else 1.0
        k_obs = float(np.clip(k_applied * float(v @ n) / planned_n, self.k_min, self.k_max))
        self._sum = self.forget * self._sum + k_obs
        self._w = self.forget * self._w + 1.0
        self.k = float((self.prior_weight * 1.0 + self._sum) / (self.prior_weight + self._w))
        self.n += 1
        self.apply(planner)


@dataclass
class ModelAdapter:
    memory: int = 9  # bounce observations kept (median -> robust to mis-detections)
    window: int = 8  # samples per deceleration fit (0.16 s at 50 Hz)
    max_resid: float = 0.35  # [m/s] RMS velocity residual above which a window is rejected
    acc_sigma: float = 0.6  # expected noise of one fitted horizontal deceleration [m/s^2]
    prior_k_sigma: float = 0.006
    prior_wind_sigma: float = 0.04  # on c = k_drag * wind
    forget: float = 0.995
    min_bounce_vz: float = 1.0
    # change detection: the identified values replace the nominal ones only beyond these
    e_tol: float = 0.06
    mu_tol: float = 0.12
    k_tol: float = 0.3  # relative
    wind_tol: float = 0.8  # [m/s]
    min_bounces: int = 3
    min_air: int = 15
    e: float = 0.0
    mu: float = 0.0
    k_drag: float = 0.0
    wind: np.ndarray = field(default_factory=lambda: np.zeros(2))
    n_bounces: int = 0
    n_air: int = 0
    _base: BallModel | None = None
    _buf: list = field(default_factory=list)
    _e_obs: list = field(default_factory=list)
    _mu_obs: list = field(default_factory=list)
    _ball_id: int = -1
    _used_bounce: int = -1
    _fitted_upto: int = 0
    _A: np.ndarray = field(default_factory=lambda: np.zeros((3, 3)))
    _b: np.ndarray = field(default_factory=lambda: np.zeros(3))

    # ------------------------------------------------------------------ public
    def reset(self, player: PhysicsPlayer, keep: bool = False) -> None:
        if self._base is None:
            self._base = player.predictor.model
        self._buf, self._ball_id, self._used_bounce, self._fitted_upto = [], -1, -1, 0
        if keep:
            return
        m = self._base
        self.e, self.mu, self.k_drag = m.bounce_e, m.bounce_mu, m.k_drag
        self.wind = np.array(m.wind[:2], float)
        self.n_bounces = self.n_air = 0
        self._e_obs, self._mu_obs = [], []
        self._A = np.zeros((3, 3))
        self._b = np.zeros(3)
        self._apply(player)

    @property
    def court_shifted(self) -> bool:
        m = self._base
        return self.n_bounces >= self.min_bounces and (
            abs(self.e - m.bounce_e) > self.e_tol or abs(self.mu - m.bounce_mu) > self.mu_tol
        )

    @property
    def air_shifted(self) -> bool:
        m = self._base
        return self.n_air >= self.min_air and (
            abs(self.k_drag - m.k_drag) > self.k_tol * m.k_drag
            or float(np.linalg.norm(self.wind - np.asarray(m.wind[:2]))) > self.wind_tol
        )

    @property
    def shifted(self) -> bool:
        return self.court_shifted or self.air_shifted

    def model(self) -> BallModel:
        """Ball model used for play: identified parameter groups replace the nominal ones only
        once they differ significantly (otherwise estimation noise would cost accuracy at home)."""
        m = self._base
        if self.court_shifted:
            m = replace(m, bounce_e=self.e, bounce_mu=self.mu)
        if self.air_shifted:
            m = replace(m, k_drag=self.k_drag, wind=(float(self.wind[0]), float(self.wind[1]), 0.0))
        return m

    def observe(self, view: PlayerView, player: PhysicsPlayer) -> None:
        if view.ball_id != self._ball_id:
            self._buf, self._ball_id, self._used_bounce, self._fitted_upto = [], view.ball_id, -1, 0
        if not view.point_live:
            return
        self._buf.append(
            (view.time, np.array(view.ball_pos, float), np.array(view.ball_vel, float))
        )
        changed = self._bounce()
        changed |= self._air()
        if changed:
            self._apply(player)

    # ------------------------------------------------------------------ internals
    def _apply(self, player: PhysicsPlayer) -> None:
        player.set_model(self.model())

    def _bounce_index(self) -> int | None:
        b = self._buf
        for j in range(max(1, self._used_bounce + 1), len(b)):
            (_, p0, v0), (_, p1, v1) = b[j - 1], b[j]
            if v0[2] < -0.3 and v1[2] > 0.3 and min(p0[2], p1[2]) < 0.2:
                return j
        return None

    def _bounce(self) -> bool:
        j = self._bounce_index()
        if j is None or j < 3 or len(self._buf) < j + 3:
            return False
        self._used_bounce = j
        r, g = 0.033, self._base.gravity
        pre, post = self._buf[j - 3 : j], self._buf[j : j + 3]
        # bounce time from the pre-bounce samples (ballistic descent to z = r)
        tbs = []
        for t, p, v in pre:
            vz = v[2]
            disc = vz * vz + 2.0 * g * max(p[2] - r, 0.0)
            tbs.append(t + (vz + np.sqrt(disc)) / g)
        tb = float(np.mean(tbs))
        if not pre[-1][0] <= tb + 0.01 or not tb <= post[0][0] + 0.01:
            return False  # not a clean floor bounce between the two samples
        vz_in = float(np.mean([v[2] - g * (tb - t) for t, _, v in pre]))
        vz_out = float(np.mean([v[2] + g * (t - tb) for t, _, v in post]))
        vt_in = np.mean([v[:2] for _, _, v in pre], axis=0)
        vt_out = np.mean([v[:2] for _, _, v in post], axis=0)
        if vz_in > -self.min_bounce_vz:
            return False
        e_obs = float(np.clip(-vz_out / vz_in, 0.4, 1.0))
        self._e_obs = [*self._e_obs, e_obs][-self.memory :]
        self.e = float(np.median([self._base.bounce_e, *self._e_obs]))
        s_in, s_out = float(np.linalg.norm(vt_in)), float(np.linalg.norm(vt_out))
        beta = self._base.bounce_beta
        if s_in > 1.0:
            mu_lb = (1.0 - beta) * s_in / ((1.0 + e_obs) * abs(vz_in))
            if s_out / s_in > beta + 0.03:  # sliding bounce -> friction observable
                mu_obs = (s_in - s_out) / ((1.0 + e_obs) * abs(vz_in))
            else:  # gripping bounce -> only a lower bound on the friction
                mu_obs = max(self.mu, mu_lb)
            self._mu_obs = [*self._mu_obs, float(np.clip(mu_obs, 0.03, 0.8))][-self.memory :]
            self.mu = float(np.median([self._base.bounce_mu, *self._mu_obs]))
        self.n_bounces += 1
        return True

    def _air(self) -> bool:
        """Fit the horizontal deceleration on bounce-free windows -> RLS on (k, k*wind)."""
        b, w = self._buf, self.window
        start = max(self._fitted_upto, 2)  # skip the samples right after the racket contact
        if len(b) - start < w:
            return False
        seg = b[start : start + w]
        self._fitted_upto = start + w
        vz = np.array([s[2][2] for s in seg])
        if np.any(np.array([s[1][2] for s in seg]) < 0.12) or np.any(np.diff(np.sign(vz)) > 0):
            return False  # touches a bounce
        t = np.array([s[0] for s in seg])
        v = np.array([s[2] for s in seg])
        tc = t - t.mean()
        acc = (tc @ (v - v.mean(axis=0))) / float(tc @ tc)
        vm = v.mean(axis=0)
        resid = v - vm - np.outer(tc, acc)
        if float(np.sqrt(np.mean(resid**2))) > self.max_resid:
            return False  # a contact (racket, net, body) inside the window
        u = np.array([vm[0] - self.wind[0], vm[1] - self.wind[1], vm[2]])
        su = float(np.linalg.norm(u))
        if su < 2.0:
            return False
        # a_h = -k |u| v_h + |u| c,   c = k * wind   (horizontal rows; Magnus mostly vertical)
        rows = np.array([[-su * vm[0], su, 0.0], [-su * vm[1], 0.0, su]])
        y = acc[:2]
        self._A = self.forget * self._A + rows.T @ rows / self.acc_sigma**2
        self._b = self.forget * self._b + rows.T @ y / self.acc_sigma**2
        theta0 = np.array([self._base.k_drag, self._base.k_drag * self._base.wind[0],
                           self._base.k_drag * self._base.wind[1]])  # fmt: skip
        P0 = np.diag(
            [1 / self.prior_k_sigma**2, 1 / self.prior_wind_sigma**2, 1 / self.prior_wind_sigma**2]
        )
        theta = np.linalg.solve(P0 + self._A, P0 @ theta0 + self._b)
        k = float(np.clip(theta[0], 0.4 * self._base.k_drag, 3.0 * self._base.k_drag))
        self.k_drag = k
        self.wind = np.clip(theta[1:] / k, -4.0, 4.0)
        self.n_air += 1
        return True


def make_adaptive(player: PhysicsPlayer, residual=None) -> PhysicsPlayer:
    """Equip a physics player with online system identification (B*), optionally combined
    with a residual landing model learned on the home court. The residual is switched off as
    soon as the identification detects a different court / air / racket, because it was
    learned for the nominal model and does not transfer (see EXPERIMENTS.md)."""
    from tennis.learning.residual import AimCorrector

    player.model_adapter = ModelAdapter()
    player.impact_adapter = ImpactAdapter()
    if residual is not None:
        corr = AimCorrector(residual)
        corr.gate = lambda: not (player.model_adapter.shifted or player.impact_adapter.shifted)
        player.aim_correction = corr
    player.reset(player.rng)
    return player
