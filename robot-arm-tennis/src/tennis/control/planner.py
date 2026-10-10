"""Hit planning: contact selection, shot solving, racket-impact inversion, IK and swing timing.

Racket impact (face normal n, racket velocity v_r, effective restitution e of the hand-held
racket, friction neglected):

    v_out = v_in + (1 + e) ((v_r - v_in) . n) n

so a desired outgoing velocity v_out fixes  n = (v_out - v_in)/|v_out - v_in|  and the racket
speed along the normal  s = v_in . n + |v_out - v_in| / (1 + e).  The racket face centre at
contact is the ball centre minus (ball radius + half face thickness) along n.

Swing (per joint, quintic): backswing pose q_b = q_c - 0.5 qd_c T  ->  contact (q_c, qd_c, 0)
-> follow-through q_f = q_c + 0.5 qd_c T. Peak acceleration is 1.5 |qd_c| / T, which sets T.
Everything is checked against joint position/velocity/acceleration limits; infeasible
contacts are rejected (robot travel time vs. ball arrival time).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, replace

import numpy as np

from tennis.control.ballistics import BallPredictor, Prediction
from tennis.sim.config import TennisConfig
from tennis.sim.kinematics import ArmKinematics
from tennis.sim.trajectory import JointTrajectory, quintic_coeffs, quintic_peaks


@dataclass(frozen=True)
class ShotSpec:
    target: tuple[float, float] = (2.0, 0.0)  # landing point in the opponent's half (x > 0)
    speed: float = 5.5  # horizontal speed of the drag-free initial guess [m/s]
    label: str = "centre"
    net_margin: float = 0.10


@dataclass(frozen=True)
class RacketCommand:
    """Direct racket parameters (used by the end-to-end RL controller instead of a ShotSpec)."""

    pitch: float  # face-normal elevation [rad]
    yaw: float  # face-normal azimuth [rad] (0 = straight down the court)
    speed: float  # racket speed along the normal [m/s]
    label: str = "rl"


@dataclass
class HitPlan:
    t_contact: float
    ball_pos: np.ndarray
    ball_vel: np.ndarray
    v_out: np.ndarray | None
    normal: np.ndarray
    racket_speed: float
    racket_pos: np.ndarray
    q_contact: np.ndarray
    qd_contact: np.ndarray
    q_back: np.ndarray
    q_follow: np.ndarray
    t_swing: float
    shot: ShotSpec | RacketCommand
    shot_info: dict = field(default_factory=dict)
    trajectory: JointTrajectory | None = None

    @property
    def t_swing_start(self) -> float:
        return self.t_contact - self.t_swing


@dataclass(frozen=True)
class PlannerConfig:
    # Racket impact model  v_out = c_t * v_in_t + (a * v_in_n + b * s) * n  (n: face normal,
    # s: racket speed along n). a = -e, b = 1 + e for an ideal collision; identified from
    # real swings in scripts/calibrate_impact.py (results/calibration.json -> "impact").
    impact_a: float = -0.689
    impact_b: float = 1.541
    impact_ct: float = 0.972
    # Slow impacts (the drop serve hits an almost stationary ball) rebound more than the
    # groundstroke-identified model predicts: b is scaled by this gain for |v_in| <= 1 m/s,
    # blending linearly to 1 at 3 m/s (scripts/calibrate_impact.py --serve).
    slow_impact_gain: float = 1.106
    face_half_thickness: float = 0.012
    rot_penalty: float = 0.0  # weight of the racket angular speed in the swing solution
    forward: tuple[float, ...] = (0.62, 0.48, 0.76)  # shoulder -> racket offsets to try
    right: tuple[float, ...] = (0.50, 0.38)
    min_swing: float = 0.14
    max_racket_speed: float = 7.0
    vel_margin: float = 0.92
    acc_margin: float = 0.92
    min_racket_height: float = 0.07
    contact_z: tuple[float, float] = (0.25, 1.25)
    preferred_z: float = 0.75
    volley_penalty: float = 0.5
    min_time: float = 0.2
    post_bounce_time: float = 0.26
    post_bounce_penalty: float = 0.6
    candidates: int = 6
    speed_backoff: tuple[float, ...] = (1.0, 0.85, 0.7)


@dataclass
class Candidate:
    t: float  # relative to now
    pos: np.ndarray
    vel: np.ndarray
    volley: bool
    score: float


class SwingPlanner:
    def __init__(
        self, cfg: TennisConfig, planner: PlannerConfig | None = None, speed_scale: float = 1.0
    ) -> None:
        self.cfg = cfg
        self.pc = planner or PlannerConfig()
        self.kin = ArmKinematics(cfg.arm, cfg.gantry)
        self.r_ball = cfg.ball.radius
        self.lo = np.concatenate(
            [[cfg.gantry.x_range[0], cfg.gantry.y_range[0]], np.radians(cfg.arm.lower_deg)]
        )
        self.hi = np.concatenate(
            [[cfg.gantry.x_range[1], cfg.gantry.y_range[1]], np.radians(cfg.arm.upper_deg)]
        )
        self.base_vmax = np.array(list(cfg.gantry.vmax) + list(cfg.arm.vmax))
        self.base_amax = np.array(list(cfg.gantry.amax) + list(cfg.arm.amax))
        self.set_speed_scale(speed_scale)

    def set_speed_scale(self, s: float) -> None:
        self.speed_scale = s
        self.vmax = self.base_vmax * s
        self.amax = self.base_amax * s * s

    # ------------------------------------------------------------------ impact model
    def _impact_b(self, v_in: np.ndarray) -> float:
        pc = self.pc
        w = min(max((3.0 - float(np.linalg.norm(v_in))) / 2.0, 0.0), 1.0)
        return pc.impact_b * (1.0 + (pc.slow_impact_gain - 1.0) * w)

    def impact_inverse(self, v_in: np.ndarray, v_out: np.ndarray) -> tuple[np.ndarray, float]:
        """Face normal and racket speed that turn v_in into v_out under the impact model."""
        pc = self.pc
        d = v_out - pc.impact_ct * v_in
        n = d / float(np.linalg.norm(d))
        return n, (float(v_out @ n) - pc.impact_a * float(v_in @ n)) / self._impact_b(v_in)

    def impact_forward(self, v_in: np.ndarray, n: np.ndarray, s: float) -> np.ndarray:
        pc = self.pc
        vn = float(v_in @ n)
        return pc.impact_ct * (v_in - vn * n) + (pc.impact_a * vn + self._impact_b(v_in) * s) * n

    @staticmethod
    def command_normal(cmd: RacketCommand) -> np.ndarray:
        cp = math.cos(cmd.pitch)
        return np.array([cp * math.cos(cmd.yaw), cp * math.sin(cmd.yaw), math.sin(cmd.pitch)])

    # ------------------------------------------------------------------ arm
    def solve_arm(
        self, p_racket: np.ndarray, n: np.ndarray, s: float, q_now: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray] | None:
        """IK at contact (trying several gantry placements) + joint velocities for v_r = s n."""
        pr = self.kin.player_to_root(p_racket)
        lo, hi = self.lo, self.hi
        options = [(f, r) for f in self.pc.forward for r in self.pc.right]
        tried = set()
        for f, r in options:
            gx = float(np.clip(pr[0] - f, lo[0], hi[0]))
            gy = float(np.clip(pr[1] + r, lo[1], hi[1]))
            key = (round(gx, 3), round(gy, 3))
            if key in tried:
                continue
            tried.add(key)
            q = self.kin.ik(pr, n, (gx, gy), q_ref=q_now)
            if q is None:
                continue
            jac = self.kin.jacobian(q)
            # Only the racket's linear velocity is prescribed (s n). Its angular velocity is
            # softly penalised: minimise sum((qd_i / vmax_i)^2) + w_rot |omega|^2 subject to
            # J_v qd = s n, so the joints share the effort and the face barely rotates
            # during the ~5 ms contact.
            jv, jw = jac[:3], jac[3:]
            h = np.diag(1.0 / self.vmax[2:] ** 2) + self.pc.rot_penalty * (jw.T @ jw)
            try:
                hinv_jvt = np.linalg.solve(h, jv.T)
                lam = np.linalg.solve(jv @ hinv_jvt, s * n)
            except np.linalg.LinAlgError:
                continue
            qd_arm = hinv_jvt @ lam
            qd = np.concatenate([[0.0, 0.0], qd_arm])
            if np.any(np.abs(qd) > self.pc.vel_margin * self.vmax):
                continue
            return q, qd
        return None

    def swing_poses(
        self, q_c: np.ndarray, qd_c: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, float] | None:
        T = max(
            self.pc.min_swing, float(np.max(1.5 * np.abs(qd_c) / (self.pc.acc_margin * self.amax)))
        )
        q_b = q_c - 0.5 * qd_c * T
        q_f = q_c + 0.5 * qd_c * T
        margin = np.concatenate([[0.0, 0.0], np.full(6, 0.02)])
        for q in (q_b, q_f):
            if np.any(q < self.lo + margin) or np.any(q > self.hi - margin):
                return None
            if self.kin.fk(q)[0][2] < self.pc.min_racket_height:
                return None
        return q_b, q_f, T

    def approach_ok(
        self, start: tuple[np.ndarray, np.ndarray, np.ndarray], q_b: np.ndarray, T: float
    ) -> bool:
        if T < 0.06:
            return bool(np.all(np.abs(start[0] - q_b) < 0.01) and np.all(np.abs(start[1]) < 0.05))
        c = quintic_coeffs(start[0], start[1], start[2], q_b, np.zeros(8), np.zeros(8), T)
        vpk, apk = quintic_peaks(c, T)
        return bool(np.all(vpk <= self.vmax * 1.02) and np.all(apk <= self.amax * 1.05))

    # ------------------------------------------------------------------ plan
    def plan(
        self,
        now: float,
        start: tuple[np.ndarray, np.ndarray, np.ndarray],
        t_rel: float,
        ball_pos: np.ndarray,
        ball_vel: np.ndarray,
        shot: ShotSpec | RacketCommand,
        predictor: BallPredictor,
    ) -> HitPlan | None:
        backoffs = self.pc.speed_backoff if isinstance(shot, ShotSpec) else (1.0,)
        for k in backoffs:
            info: dict = {}
            v_out = None
            if isinstance(shot, ShotSpec):
                sol = predictor.solve_shot(
                    ball_pos, shot.target, shot.speed * k, net_margin=shot.net_margin
                )
                if sol is None:
                    continue
                v_out, info = sol
                n, s = self.impact_inverse(ball_vel, v_out)
            else:
                n = self.command_normal(shot)
                s = float(shot.speed)
            if n[0] < 0.25 or not 0.2 <= s <= self.pc.max_racket_speed * self.speed_scale:
                continue
            p_racket = ball_pos - (self.r_ball + self.pc.face_half_thickness) * n
            arm = self.solve_arm(p_racket, n, s, start[0])
            if arm is None:
                continue
            q_c, qd_c = arm
            sw = self.swing_poses(q_c, qd_c)
            if sw is None:
                continue
            q_b, q_f, T = sw
            if not self.approach_ok(start, q_b, t_rel - T):
                continue
            plan = HitPlan(
                t_contact=now + t_rel,
                ball_pos=np.asarray(ball_pos, float),
                ball_vel=np.asarray(ball_vel, float),
                v_out=v_out,
                normal=n,
                racket_speed=s,
                racket_pos=p_racket,
                q_contact=q_c,
                qd_contact=qd_c,
                q_back=q_b,
                q_follow=q_f,
                t_swing=T,
                shot=shot
                if k == 1.0 or not isinstance(shot, ShotSpec)
                else replace(shot, speed=shot.speed * k),
                shot_info=info,
            )
            plan.trajectory = self.build(plan, now, start)
            return plan
        return None

    def build(
        self, plan: HitPlan, now: float, start: tuple[np.ndarray, np.ndarray, np.ndarray]
    ) -> JointTrajectory:
        traj = JointTrajectory(start[0], now)
        z = np.zeros(8)
        T_pre = plan.t_swing_start - now
        if T_pre > 1e-3:
            traj.append(plan.q_back, z, T_pre, label="prepare", start=start, t_start=now)
            traj.append(plan.q_contact, plan.qd_contact, plan.t_swing, label="swing")
        else:  # already in the swing window: go straight for the contact state
            traj.append(
                plan.q_contact,
                plan.qd_contact,
                plan.t_contact - now,
                label="swing",
                start=start,
                t_start=now,
            )
        traj.append(plan.q_follow, z, plan.t_swing, label="follow")
        return traj

    def move_to(
        self,
        now: float,
        start: tuple[np.ndarray, np.ndarray, np.ndarray],
        q_goal: np.ndarray,
        label: str = "recover",
    ) -> JointTrajectory:
        """Smooth limit-respecting move to a rest pose."""
        dq = np.abs(q_goal - start[0])
        T = float(
            np.max(
                np.maximum(1.875 * dq / (0.8 * self.vmax), np.sqrt(5.8 * dq / (0.8 * self.amax)))
            )
        )
        T = max(T + 0.4 * float(np.max(np.abs(start[1]) / self.amax)), 0.25)
        traj = JointTrajectory(start[0], now)
        traj.append(q_goal, np.zeros(8), T, label=label, start=start, t_start=now)
        return traj


def contact_candidates(
    pred: Prediction,
    pc: PlannerConfig,
    kin: ArmKinematics,
    lo: np.ndarray,
    hi: np.ndarray,
    after_bounce_only: bool,
    bounced_already: bool,
) -> list[Candidate]:
    """Reachable, legal interception points on a predicted incoming trajectory (sorted)."""
    out: list[Candidate] = []
    x_lo = lo[0] + min(pc.forward) - 0.15 - kin.root_x
    x_hi = hi[0] + max(pc.forward) + 0.05 - kin.root_x
    y_lo = lo[1] - max(pc.right) - 0.1
    y_hi = hi[1] - min(pc.right) + 0.1
    z_lo, z_hi = pc.contact_z
    nb = pred.n_bounce_before
    bounces = pred.bounces()
    t_b = bounces[0].t if bounces and not bounced_already else None
    for i in range(0, len(pred.t), 2):
        t = float(pred.t[i])
        if t < pc.min_time:
            continue
        x, y, z = pred.pos[i]
        if x > -0.15:
            continue
        b = int(nb[i]) + (1 if bounced_already else 0)
        if b >= 2 or (after_bounce_only and b < 1):
            continue
        if not (z_lo <= z <= z_hi and x_lo <= x <= x_hi and y_lo <= y <= y_hi):
            continue
        volley = b == 0
        score = (
            abs(z - pc.preferred_z)
            + (pc.volley_penalty if volley else 0.0)
            + 0.3 * max(0.0, 0.45 - t)
        )
        if t_b is not None and t > t_b and t - t_b < pc.post_bounce_time:
            # The swing would be committed before the bounce is observed: bounces are
            # irregular, so prefer contacts that can be re-planned after seeing the bounce.
            score += pc.post_bounce_penalty
        out.append(Candidate(t, pred.pos[i].copy(), pred.vel[i].copy(), volley, score))
    out.sort(key=lambda c: c.score)
    return out
