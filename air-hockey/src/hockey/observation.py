"""Observation layer: player-frame views, sensor noise and RL observation vectors.

Every controller works in its own *player frame*: its goal is at x = -L/2 and it plays
towards +x. For the away robot the world is mirrored in x (y is unchanged), so the same
controller code drives either robot.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

from hockey.simulation.config import SimConfig
from hockey.simulation.mjcf import joint_ranges
from hockey.simulation.physics import SimState
from hockey.simulation.robot import AxisLimits

POS_SCALE = np.array([1.0, 0.5])  # half table size (L/2, W/2) for the default table
VEL_SCALE = 5.0
OBS_CLIP = 5.0


@dataclass(frozen=True)
class Geometry:
    """Static geometry/limits a controller may know about (player frame)."""

    length: float
    width: float
    goal_width: float
    puck_radius: float
    mallet_radius: float
    x_range: tuple[float, float]
    y_range: tuple[float, float]
    vmax: tuple[float, float]
    amax: tuple[float, float]
    control_dt: float
    home_x: float

    @property
    def half_length(self) -> float:
        return self.length / 2

    @property
    def half_width(self) -> float:
        return self.width / 2

    def axis_limits(self, speed_scale: float = 1.0) -> tuple[AxisLimits, AxisLimits]:
        return (
            AxisLimits(*self.x_range, self.vmax[0] * speed_scale, self.amax[0] * speed_scale),
            AxisLimits(*self.y_range, self.vmax[1] * speed_scale, self.amax[1] * speed_scale),
        )

    @staticmethod
    def from_sim_config(cfg: SimConfig, player: str = "home") -> Geometry:
        robot = cfg.robot if player == "home" else cfg.opponent_robot
        xr, yr = joint_ranges(cfg, "home")  # player frame == home frame
        return Geometry(
            length=cfg.table.length,
            width=cfg.table.width,
            goal_width=cfg.table.goal_width,
            puck_radius=cfg.puck.radius,
            mallet_radius=cfg.mallet.radius,
            x_range=xr,
            y_range=yr,
            vmax=robot.vmax,
            amax=robot.amax,
            control_dt=cfg.physics.control_dt,
            home_x=-cfg.table.length / 2 + robot.home_offset,
        )


@dataclass
class PlayerView:
    """Snapshot of the game as seen by one player (player frame)."""

    time: float
    puck_pos: np.ndarray
    puck_vel: np.ndarray
    own_pos: np.ndarray
    own_vel: np.ndarray
    opp_pos: np.ndarray
    opp_vel: np.ndarray
    score_own: int = 0
    score_opp: int = 0
    time_remaining: float = 0.0


_MIRROR = np.array([-1.0, 1.0])


def player_view(
    state: SimState,
    player: str,
    score: tuple[int, int] = (0, 0),
    time_remaining: float = 0.0,
) -> PlayerView:
    """``score`` is (home, away) in world terms."""
    if player == "home":
        return PlayerView(
            time=state.time,
            puck_pos=state.puck_pos.copy(),
            puck_vel=state.puck_vel.copy(),
            own_pos=state.home_pos.copy(),
            own_vel=state.home_vel.copy(),
            opp_pos=state.away_pos.copy(),
            opp_vel=state.away_vel.copy(),
            score_own=score[0],
            score_opp=score[1],
            time_remaining=time_remaining,
        )
    return PlayerView(
        time=state.time,
        puck_pos=state.puck_pos * _MIRROR,
        puck_vel=state.puck_vel * _MIRROR,
        own_pos=state.away_pos * _MIRROR,
        own_vel=state.away_vel * _MIRROR,
        opp_pos=state.home_pos * _MIRROR,
        opp_vel=state.home_vel * _MIRROR,
        score_own=score[1],
        score_opp=score[0],
        time_remaining=time_remaining,
    )


def to_world(player: str, xy: np.ndarray | tuple[float, float]) -> np.ndarray:
    xy = np.asarray(xy, dtype=float)
    return xy.copy() if player == "home" else xy * _MIRROR


@dataclass(frozen=True)
class SensorNoise:
    pos_std: float = 0.0
    vel_std: float = 0.0


def apply_noise(view: PlayerView, noise: SensorNoise, rng: np.random.Generator) -> PlayerView:
    """Gaussian measurement noise on the puck and opponent (own encoders are exact)."""
    if noise.pos_std <= 0 and noise.vel_std <= 0:
        return view
    return replace(
        view,
        puck_pos=view.puck_pos + rng.normal(0.0, noise.pos_std, 2),
        puck_vel=view.puck_vel + rng.normal(0.0, noise.vel_std, 2),
        opp_pos=view.opp_pos + rng.normal(0.0, noise.pos_std, 2),
        opp_vel=view.opp_vel + rng.normal(0.0, noise.vel_std, 2),
    )


OBS_NAMES = (
    "puck_x",
    "puck_y",
    "puck_vx",
    "puck_vy",
    "own_x",
    "own_y",
    "own_vx",
    "own_vy",
    "opp_x",
    "opp_y",
    "opp_vx",
    "opp_vy",
    "rel_x",
    "rel_y",
    "side_clock",
)
OBS_DIM = len(OBS_NAMES)
SIDE_CLOCK_NORM = 7.0  # possession timeout of the rules [s]


class SideClock:
    """Game-state feature: time since the puck entered its current half (possession clock).

    Reset on face-offs (detected as a puck jump) so that training episodes and full matches
    produce the same statistics.
    """

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self._side = 0
        self._since = 0.0
        self._last_pos: np.ndarray | None = None
        self.value = 0.0

    def update(self, view: PlayerView) -> float:
        side = -1 if view.puck_pos[0] < 0 else 1
        jumped = (
            self._last_pos is not None and float(np.hypot(*(view.puck_pos - self._last_pos))) > 0.3
        )
        if side != self._side or jumped or self._last_pos is None:
            self._side = side
            self._since = view.time
        self._last_pos = view.puck_pos.copy()
        self.value = view.time - self._since
        return self.value


def encode_observation(view: PlayerView, geom: Geometry, side_time: float = 0.0) -> np.ndarray:
    """Normalised state vector (float32, length ``OBS_DIM``)."""
    scale = np.array([geom.half_length, geom.half_width])
    obs = np.concatenate(
        [
            view.puck_pos / scale,
            view.puck_vel / VEL_SCALE,
            view.own_pos / scale,
            view.own_vel / VEL_SCALE,
            view.opp_pos / scale,
            view.opp_vel / VEL_SCALE,
            (view.puck_pos - view.own_pos) / scale,
            [min(side_time / SIDE_CLOCK_NORM, 1.5)],
        ]
    )
    return np.clip(obs, -OBS_CLIP, OBS_CLIP).astype(np.float32)
