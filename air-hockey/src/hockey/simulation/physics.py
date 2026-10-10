"""MuJoCo air-hockey simulation: stepping, contact events, goals, parameter changes."""

from __future__ import annotations

from dataclasses import dataclass, field

import mujoco
import numpy as np

from hockey.simulation.config import RobotConfig, SimConfig
from hockey.simulation.mjcf import (
    PLAYERS,
    build_mjcf,
    contact_solref,
    joint_ranges,
    wall_geoms,
)
from hockey.simulation.robot import AxisLimits, MotionController


@dataclass
class SimState:
    """World-frame snapshot (copies, safe to keep)."""

    time: float
    puck_pos: np.ndarray
    puck_vel: np.ndarray
    home_pos: np.ndarray
    home_vel: np.ndarray
    away_pos: np.ndarray
    away_vel: np.ndarray

    def player_pos(self, player: str) -> np.ndarray:
        return self.home_pos if player == "home" else self.away_pos

    def player_vel(self, player: str) -> np.ndarray:
        return self.home_vel if player == "home" else self.away_vel


@dataclass
class ContactEvent:
    kind: str  # "home" | "away" (mallet hit) | "wall"
    time: float
    puck_pos: tuple[float, float]
    puck_vel_before: tuple[float, float]


@dataclass
class StepResult:
    events: list[ContactEvent] = field(default_factory=list)
    goal: str | None = None  # player who SCORED ("home" scores into the away goal)
    out_of_bounds: bool = False
    target_clipped: dict[str, bool] = field(default_factory=dict)
    mallet_contact: bool = False  # any puck-mallet contact during this control step


@dataclass(frozen=True)
class PhysicsParams:
    """Parameters that may be randomised per episode (domain randomisation)."""

    puck_decay: float
    wall_restitution: float
    mallet_restitution: float
    home_speed_scale: float = 1.0
    away_speed_scale: float = 1.0


def _robot_limits(cfg: SimConfig, player: str, robot: RobotConfig) -> tuple[AxisLimits, AxisLimits]:
    (xlo, xhi), (ylo, yhi) = joint_ranges(cfg, player)
    return (
        AxisLimits(xlo, xhi, robot.vmax[0], robot.amax[0]),
        AxisLimits(ylo, yhi, robot.vmax[1], robot.amax[1]),
    )


class HockeySim:
    """Two gantry robots + puck on a MuJoCo air-hockey table.

    Robots are only moved through joint actuators (velocity servos). ``qpos`` is written
    directly exclusively during ``reset`` (initial conditions).
    """

    def __init__(self, cfg: SimConfig | None = None) -> None:
        self.cfg = cfg or SimConfig()
        self.model = mujoco.MjModel.from_xml_string(build_mjcf(self.cfg))
        self.data = mujoco.MjData(self.model)
        m = self.model
        jid = lambda n: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, n)  # noqa: E731
        gid = lambda n: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, n)  # noqa: E731
        aid = lambda n: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_ACTUATOR, n)  # noqa: E731

        self._puck_q = (int(m.jnt_qposadr[jid("puck_x")]), int(m.jnt_qposadr[jid("puck_y")]))
        self._puck_v = (int(m.jnt_dofadr[jid("puck_x")]), int(m.jnt_dofadr[jid("puck_y")]))
        self._q = {
            p: (int(m.jnt_qposadr[jid(f"{p}_x")]), int(m.jnt_qposadr[jid(f"{p}_y")]))
            for p in PLAYERS
        }
        self._v = {
            p: (int(m.jnt_dofadr[jid(f"{p}_x")]), int(m.jnt_dofadr[jid(f"{p}_y")])) for p in PLAYERS
        }
        self._act = {p: (aid(f"{p}_x_servo"), aid(f"{p}_y_servo")) for p in PLAYERS}
        self._puck_geom = gid("puck_col")
        self._mallet_geom = {gid(f"{p}_mallet_col"): p for p in PLAYERS}
        self._wall_geoms = {gid(name) for name, *_ in wall_geoms(self.cfg)}
        self._wall_pairs = [
            mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_PAIR, f"puck_{name}")
            for name, *_ in wall_geoms(self.cfg)
            if not name.startswith("goal_")
        ]
        self._mallet_pairs = [
            mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_PAIR, f"puck_{p}_mallet") for p in PLAYERS
        ]
        self._puck_dofs = list(self._puck_v)
        self._base_force = {p: m.actuator_forcerange[list(self._act[p])].copy() for p in PLAYERS}
        self._base_ctrl = {p: m.actuator_ctrlrange[list(self._act[p])].copy() for p in PLAYERS}

        robots = {"home": self.cfg.robot, "away": self.cfg.opponent_robot}
        self.limits = {p: _robot_limits(self.cfg, p, robots[p]) for p in PLAYERS}
        self.motion = {
            p: MotionController(*self.limits[p], robots[p].position_gain, robots[p].brake_ratio)
            for p in PLAYERS
        }
        self.targets = {p: self.home_position(p) for p in PLAYERS}
        self._switch: dict[str, tuple[float, np.ndarray] | None] = {p: None for p in PLAYERS}
        self.frame_skip = self.cfg.frame_skip
        self.dt = self.cfg.physics.timestep
        self.motion_steps = max(1, min(self.cfg.physics.motion_steps, self.frame_skip))
        if self.frame_skip % self.motion_steps:
            raise ValueError("control period must be a multiple of the motion-controller period")
        self.control_dt = self.frame_skip * self.dt
        self._in_contact: dict[str, bool] = {"home": False, "away": False, "wall": False}
        self.params = PhysicsParams(
            puck_decay=self.cfg.puck.decay,
            wall_restitution=self.cfg.contact.wall_restitution,
            mallet_restitution=self.cfg.contact.mallet_restitution,
        )
        self.reset()

    # ------------------------------------------------------------------ geometry
    def home_position(self, player: str) -> np.ndarray:
        L2 = self.cfg.table.length / 2
        offset = (self.cfg.robot if player == "home" else self.cfg.opponent_robot).home_offset
        x = -L2 + offset if player == "home" else L2 - offset
        return np.array([x, 0.0])

    def actual_limits(self, player: str) -> tuple[AxisLimits, AxisLimits]:
        return self.motion[player].limits  # type: ignore[return-value]

    # ------------------------------------------------------------------ parameters
    def set_params(self, params: PhysicsParams) -> None:
        m = self.model
        damping = params.puck_decay * self.cfg.puck.mass
        for dof in self._puck_dofs:
            m.dof_damping[dof] = damping
        freq = self.cfg.contact.frequency
        ref_wall = contact_solref(params.wall_restitution, freq)
        ref_mallet = contact_solref(params.mallet_restitution, freq)
        for pid in self._wall_pairs:
            m.pair_solref[pid] = ref_wall
        for pid in self._mallet_pairs:
            m.pair_solref[pid] = ref_mallet
        for player, scale in (("home", params.home_speed_scale), ("away", params.away_speed_scale)):
            acts = list(self._act[player])
            m.actuator_forcerange[acts] = self._base_force[player] * scale
            m.actuator_ctrlrange[acts] = self._base_ctrl[player] * scale
            self.motion[player].set_speed_scale(scale)
        self.params = params

    # ------------------------------------------------------------------ state
    def state(self) -> SimState:
        q, v = self.data.qpos, self.data.qvel
        return SimState(
            time=float(self.data.time),
            puck_pos=np.array([q[self._puck_q[0]], q[self._puck_q[1]]]),
            puck_vel=np.array([v[self._puck_v[0]], v[self._puck_v[1]]]),
            home_pos=np.array([q[self._q["home"][0]], q[self._q["home"][1]]]),
            home_vel=np.array([v[self._v["home"][0]], v[self._v["home"][1]]]),
            away_pos=np.array([q[self._q["away"][0]], q[self._q["away"][1]]]),
            away_vel=np.array([v[self._v["away"][0]], v[self._v["away"][1]]]),
        )

    @property
    def time(self) -> float:
        return float(self.data.time)

    # ------------------------------------------------------------------ reset
    def reset(
        self,
        puck_pos: tuple[float, float] | np.ndarray = (0.0, 0.0),
        puck_vel: tuple[float, float] | np.ndarray = (0.0, 0.0),
        home_pos: tuple[float, float] | np.ndarray | None = None,
        away_pos: tuple[float, float] | np.ndarray | None = None,
        keep_time: bool = False,
    ) -> None:
        t0 = float(self.data.time) if keep_time else 0.0
        mujoco.mj_resetData(self.model, self.data)
        self.data.time = t0
        q, v = self.data.qpos, self.data.qvel
        q[self._puck_q[0]], q[self._puck_q[1]] = float(puck_pos[0]), float(puck_pos[1])
        v[self._puck_v[0]], v[self._puck_v[1]] = float(puck_vel[0]), float(puck_vel[1])
        for player, pos in (("home", home_pos), ("away", away_pos)):
            p = self.home_position(player) if pos is None else np.asarray(pos, dtype=float)
            p = np.array(self.motion[player].clip_target(float(p[0]), float(p[1])))
            q[self._q[player][0]], q[self._q[player][1]] = p
            self.targets[player] = p.copy()
            self._switch[player] = None
            self.motion[player].reset()
        self._in_contact = {"home": False, "away": False, "wall": False}
        mujoco.mj_forward(self.model, self.data)

    # ------------------------------------------------------------------ control
    def set_target(
        self,
        player: str,
        target: tuple[float, float] | np.ndarray,
        switch_target: tuple[float, float] | np.ndarray | None = None,
        switch_time: float | None = None,
    ) -> None:
        """Set the mallet target; optionally switch to ``switch_target`` at sim time
        ``switch_time`` (time-stamped command executed by the 1 kHz low-level loop)."""
        self.targets[player] = np.asarray(target, dtype=float)
        if switch_target is not None and switch_time is not None:
            self._switch[player] = (float(switch_time), np.asarray(switch_target, dtype=float))
        else:
            self._switch[player] = None

    def step(self) -> StepResult:
        """Advance one control period (``frame_skip`` physics steps).

        The motion controllers (trajectory generators) run every ``motion_steps`` physics
        steps (250 Hz); the MuJoCo velocity servos act at every physics step (1 kHz).
        Contacts and goals are checked after every block (contacts last longer than a block).
        """
        m, d = self.model, self.data
        q, v, ctrl = d.qpos, d.qvel, d.ctrl
        L2 = self.cfg.table.length / 2
        W2 = self.cfg.table.width / 2
        rp = self.cfg.puck.radius
        g2 = self.cfg.table.goal_width / 2
        x_lim = L2 + self.cfg.table.goal_depth + self.cfg.table.wall_thickness + 0.05
        y_lim = W2 + self.cfg.table.wall_thickness + 0.05
        result = StepResult()
        px_i, py_i = self._puck_q
        pvx_i, pvy_i = self._puck_v
        block = self.motion_steps
        block_dt = block * self.dt
        clipped = {"home": False, "away": False}
        for _ in range(self.frame_skip // block):
            for player in PLAYERS:
                sw = self._switch[player]
                if sw is not None and d.time >= sw[0] - 1e-9:
                    self.targets[player] = sw[1]
                    self._switch[player] = None
                qi = self._q[player]
                tx, ty = self.targets[player]
                mc = self.motion[player]
                ux, uy = mc.command(tx, ty, q[qi[0]], q[qi[1]], block_dt)
                clipped[player] = clipped[player] or mc.clipped_last
                ai = self._act[player]
                ctrl[ai[0]] = ux
                ctrl[ai[1]] = uy
            vel_before = (float(v[pvx_i]), float(v[pvy_i]))
            mujoco.mj_step(m, d, block)
            self._detect_contacts(result, vel_before)
            px, py = float(q[px_i]), float(q[py_i])
            if px < -L2 - rp or px > L2 + rp:
                if abs(py) <= g2 + rp:
                    result.goal = "away" if px < 0 else "home"
                else:
                    result.out_of_bounds = True
                break
            if abs(px) > x_lim or abs(py) > y_lim or not np.isfinite(px + py):
                result.out_of_bounds = True
                break
        result.target_clipped = clipped
        return result

    def _detect_contacts(self, result: StepResult, vel_before: tuple[float, float]) -> None:
        d = self.data
        touching = {"home": False, "away": False, "wall": False}
        ncon = d.ncon
        if ncon:
            geoms = d.contact.geom[:ncon]
            for g1, g2 in geoms:
                other = g2 if g1 == self._puck_geom else g1 if g2 == self._puck_geom else -1
                if other < 0:
                    continue
                if other in self._mallet_geom:
                    touching[self._mallet_geom[other]] = True
                    result.mallet_contact = True
                elif other in self._wall_geoms:
                    touching["wall"] = True
        for kind, now in touching.items():
            if now and not self._in_contact[kind]:
                q = d.qpos
                result.events.append(
                    ContactEvent(
                        kind=kind,
                        time=float(d.time),
                        puck_pos=(float(q[self._puck_q[0]]), float(q[self._puck_q[1]])),
                        puck_vel_before=vel_before,
                    )
                )
            self._in_contact[kind] = now
