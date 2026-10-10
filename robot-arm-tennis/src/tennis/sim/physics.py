"""MuJoCo tennis simulation: servo tracking of joint trajectories, contact events."""

from __future__ import annotations

from dataclasses import dataclass, field

import mujoco
import numpy as np

from tennis.sim.config import TennisConfig
from tennis.sim.kinematics import ArmKinematics, world_to_player
from tennis.sim.mjcf import JOINTS, PLAYERS, body_collision_geoms, build_mjcf, contact_solref
from tennis.sim.trajectory import JointTrajectory

NOMINAL_Q = np.array([0.6, 0.4, -0.5, 0.4, -1.3, 0.0, 0.8, 0.0])  # pose used to size the servos


@dataclass
class ContactEvent:
    kind: str  # "floor" | "net" | "racket_home" | "racket_away" | "body_home" | "body_away"
    time: float
    ball_pos: np.ndarray  # world frame
    ball_vel: np.ndarray  # world frame, just before the contact


@dataclass
class StepResult:
    events: list[ContactEvent] = field(default_factory=list)
    lost: bool = False  # ball left the arena / numerical failure


@dataclass
class TennisState:
    time: float
    ball_pos: np.ndarray
    ball_vel: np.ndarray
    ball_spin: np.ndarray
    q: dict[str, np.ndarray]
    qd: dict[str, np.ndarray]
    racket_pos: dict[str, np.ndarray]
    racket_normal: dict[str, np.ndarray]


@dataclass(frozen=True)
class PhysicsParams:
    """Per-episode physical parameters (domain randomisation / unknown conditions)."""

    court_restitution: float = 0.75
    court_friction: float = 0.6
    racket_restitution: float = 0.85
    racket_friction: float = 0.4
    drag_scale: float = 1.0
    wind: tuple[float, float, float] = (0.0, 0.0, 0.0)
    speed_scale: tuple[float, float] = (1.0, 1.0)  # (home, away) robot strength


class TennisSim:
    def __init__(self, cfg: TennisConfig | None = None) -> None:
        self.cfg = cfg or TennisConfig()
        self.model = mujoco.MjModel.from_xml_string(build_mjcf(self.cfg))
        self.data = mujoco.MjData(self.model)
        m = self.model
        name = lambda t, n: mujoco.mj_name2id(m, t, n)  # noqa: E731
        J, G, A = mujoco.mjtObj.mjOBJ_JOINT, mujoco.mjtObj.mjOBJ_GEOM, mujoco.mjtObj.mjOBJ_ACTUATOR
        bj = name(J, "ball")
        self._bq = int(m.jnt_qposadr[bj])
        self._bv = int(m.jnt_dofadr[bj])
        self._ball_body = name(mujoco.mjtObj.mjOBJ_BODY, "ball")
        self._qadr = {
            p: np.array([m.jnt_qposadr[name(J, f"{p}_{j}")] for j in JOINTS]) for p in PLAYERS
        }
        self._vadr = {
            p: np.array([m.jnt_dofadr[name(J, f"{p}_{j}")] for j in JOINTS]) for p in PLAYERS
        }
        self._act = {p: np.array([name(A, f"{p}_a_{j}") for j in JOINTS]) for p in PLAYERS}
        self._site = {p: name(mujoco.mjtObj.mjOBJ_SITE, f"{p}_racket") for p in PLAYERS}
        self._ball_geom = name(G, "ball")
        kinds = {name(G, "floor"): "floor", name(G, "net"): "net"}
        for p in PLAYERS:
            kinds[name(G, f"{p}_racket_col")] = f"racket_{p}"
            for g in body_collision_geoms(p):
                kinds[name(G, g)] = f"body_{p}"
        self._kinds = kinds
        P = mujoco.mjtObj.mjOBJ_PAIR
        self._pairs = {
            "floor": [name(P, "ball_floor")],
            "racket": [name(P, f"ball_{p}_racket") for p in PLAYERS],
        }
        self.kin = ArmKinematics(self.cfg.arm, self.cfg.gantry)
        self.dt = self.cfg.physics.timestep
        self.servo_steps = self.cfg.physics.servo_steps
        self.frame_skip = self.cfg.frame_skip
        self.control_dt = self.frame_skip * self.dt
        self._setup_servos()
        self.traj = {p: JointTrajectory.hold(NOMINAL_Q.copy()) for p in PLAYERS}
        self._touch = dict.fromkeys(set(kinds.values()), False)
        self.params = PhysicsParams()
        self.set_params(self.params)
        self.reset()

    # ------------------------------------------------------------------ servos
    def _setup_servos(self) -> None:
        m, d = self.model, self.data
        arm, g = self.cfg.arm, self.cfg.gantry
        for p in PLAYERS:
            d.qpos[self._qadr[p]] = NOMINAL_Q
        mujoco.mj_forward(m, d)
        full = np.zeros((m.nv, m.nv))
        try:  # MuJoCo >= 3.4 signature
            mujoco.mj_fullM(m, d, full)
        except TypeError:
            mujoco.mj_fullM(m, full, d.qM)
        self._kvkp, self._inv_w2, self._base_force = {}, {}, {}
        omega = np.array([g.servo_bandwidth] * 2 + [arm.servo_bandwidth] * 6)
        amax = np.array(list(g.amax) + list(arm.amax))
        margin = np.array([g.force_margin] * 2 + [arm.torque_margin] * 6)
        for p in PLAYERS:
            idx = self._vadr[p]
            mii = full[idx, idx]
            kp, kv = mii * omega**2, 2.0 * mii * omega
            for k, a in enumerate(self._act[p]):
                m.actuator_gainprm[a, 0] = kp[k]
                m.actuator_biasprm[a, 1] = -kp[k]
                m.actuator_biasprm[a, 2] = -kv[k]
            self._kvkp[p] = kv / kp
            self._inv_w2[p] = 1.0 / omega**2
            # Motor sizing: torque for the rated accelerations including inertial coupling
            # between joints (row sums of |M|), times a margin.
            coupled = np.abs(full[np.ix_(idx, idx)]) @ amax
            self._base_force[p] = np.maximum(margin * coupled, 5.0)
            m.actuator_forcerange[self._act[p]] = np.column_stack(
                [-self._base_force[p], self._base_force[p]]
            )

    def set_params(self, params: PhysicsParams) -> None:
        m, k = self.model, self.cfg.contact
        freq = k.frequency
        m.pair_solref[self._pairs["floor"][0]] = contact_solref(params.court_restitution, freq)
        m.pair_friction[self._pairs["floor"][0], :2] = params.court_friction
        for pid in self._pairs["racket"]:
            m.pair_solref[pid] = contact_solref(params.racket_restitution, k.racket_frequency)
            m.pair_friction[pid, :2] = params.racket_friction
        m.geom_fluid[self._ball_geom, 1] = (
            self.cfg.ball.fluidcoef[0] * params.drag_scale
        )  # blunt drag
        m.opt.wind[:] = params.wind
        for p, s in zip(PLAYERS, params.speed_scale, strict=True):
            f = self._base_force[p] * s * s  # same trajectories scaled in time need ~s^2 force
            m.actuator_forcerange[self._act[p]] = np.column_stack([-f, f])
        self.params = params

    # ------------------------------------------------------------------ state
    @property
    def time(self) -> float:
        return float(self.data.time)

    def ball(self) -> tuple[np.ndarray, np.ndarray]:
        q, v = self.data.qpos, self.data.qvel
        return q[self._bq : self._bq + 3].copy(), v[self._bv : self._bv + 3].copy()

    def state(self) -> TennisState:
        d = self.data
        pos, vel = self.ball()
        rot = d.xmat[self._ball_body].reshape(3, 3)
        spin = rot @ d.qvel[self._bv + 3 : self._bv + 6]
        return TennisState(
            time=float(d.time),
            ball_pos=pos,
            ball_vel=vel,
            ball_spin=spin,
            q={p: d.qpos[self._qadr[p]].copy() for p in PLAYERS},
            qd={p: d.qvel[self._vadr[p]].copy() for p in PLAYERS},
            racket_pos={p: d.site_xpos[self._site[p]].copy() for p in PLAYERS},
            racket_normal={
                p: d.site_xmat[self._site[p]].reshape(3, 3)[:, 2].copy() for p in PLAYERS
            },
        )

    # ------------------------------------------------------------------ reset / control
    def reset(
        self,
        ball_pos=(0.0, 0.0, 1.0),
        ball_vel=(0.0, 0.0, 0.0),
        ball_spin=(0.0, 0.0, 0.0),
        robot_q: dict[str, np.ndarray] | None = None,
        keep_time: bool = False,
    ) -> None:
        m, d = self.model, self.data
        t0 = float(d.time) if keep_time else 0.0
        mujoco.mj_resetData(m, d)
        d.time = t0
        d.qpos[self._bq : self._bq + 3] = ball_pos
        d.qpos[self._bq + 3 : self._bq + 7] = (1.0, 0.0, 0.0, 0.0)
        d.qvel[self._bv : self._bv + 3] = ball_vel
        d.qvel[self._bv + 3 : self._bv + 6] = ball_spin
        for p in PLAYERS:
            q = NOMINAL_Q if robot_q is None or p not in robot_q else np.asarray(robot_q[p], float)
            d.qpos[self._qadr[p]] = q
            self.traj[p] = JointTrajectory.hold(q, t0)
        self._touch = dict.fromkeys(self._touch, False)
        mujoco.mj_forward(m, d)

    def place_ball(self, pos, vel, spin=(0.0, 0.0, 0.0)) -> None:
        """Re-place only the ball (point start); robots keep their state."""
        d = self.data
        d.qpos[self._bq : self._bq + 3] = pos
        d.qpos[self._bq + 3 : self._bq + 7] = (1.0, 0.0, 0.0, 0.0)
        d.qvel[self._bv : self._bv + 3] = vel
        d.qvel[self._bv + 3 : self._bv + 6] = spin
        self._touch = dict.fromkeys(self._touch, False)
        mujoco.mj_forward(self.model, d)

    def set_trajectory(self, player: str, traj: JointTrajectory) -> None:
        self.traj[player] = traj

    def step(self) -> StepResult:
        """Advance one control period; servo set-points follow the trajectories at 500 Hz."""
        m, d = self.model, self.data
        res = StepResult()
        block = self.servo_steps
        half = 0.5 * block * self.dt
        ctrl = d.ctrl
        bv = self._bv
        for _ in range(self.frame_skip // block):
            t_mid = d.time + half
            for p in PLAYERS:
                q, v, a = self.traj[p].sample(t_mid)
                ctrl[self._act[p]] = q + self._kvkp[p] * v + self._inv_w2[p] * a
            vel_before = d.qvel[bv : bv + 3].copy()
            mujoco.mj_step(m, d, block)
            self._detect(res, vel_before)
            z = d.qpos[self._bq + 2]
            x, y = d.qpos[self._bq], d.qpos[self._bq + 1]
            if not np.isfinite(x + y + z) or abs(x) > 12 or abs(y) > 8 or z < -0.5:
                res.lost = True
                break
        return res

    def _detect(self, res: StepResult, vel_before: np.ndarray) -> None:
        d = self.data
        touching = dict.fromkeys(self._touch, False)
        bg = self._ball_geom
        for g1, g2 in d.contact.geom[: d.ncon]:
            other = g2 if g1 == bg else g1 if g2 == bg else -1
            kind = self._kinds.get(int(other))
            if kind is not None:
                touching[kind] = True
        for kind, now in touching.items():
            if now and not self._touch[kind]:
                pos, _ = self.ball()
                res.events.append(ContactEvent(kind, float(d.time), pos, vel_before))
            self._touch[kind] = now

    # ------------------------------------------------------------------ helpers
    def player_ball(self, player: str) -> tuple[np.ndarray, np.ndarray]:
        pos, vel = self.ball()
        return world_to_player(player, pos), world_to_player(player, vel)
