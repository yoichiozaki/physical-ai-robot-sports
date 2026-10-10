"""Configuration of the tennis world (SI units: m, kg, s, rad).

Frames
------
World: origin at the centre of the net on the court surface, x along the court
(home baseline at x = -L/2, away baseline at x = +L/2), y across, z up.
Each robot has a *root frame*: home root at (-root_x, 0, 0) facing +x; the away root at
(+root_x, 0, 0) is rotated by 180 deg about z, so both robots are identical in their
root frames. Controllers work in the *player frame* (world frame for home, world frame
rotated by 180 deg for away): own court is x < 0, the opponent's court x > 0.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class CourtConfig:
    length: float = 6.0  # baseline to baseline (scaled "mini tennis" court for arms)
    width: float = 2.4  # singles sidelines
    net_height: float = 0.35
    net_half_span: float = 1.45  # posts at |y| = this
    line_width: float = 0.04


@dataclass(frozen=True)
class BallConfig:
    radius: float = 0.033  # tennis ball
    mass: float = 0.057
    # MuJoCo ellipsoid fluid model coefficients (blunt drag, slender drag, angular drag,
    # Kutta lift, Magnus lift). For a sphere: drag force = blunt * rho * pi r^2 |v| v.
    fluidcoef: tuple[float, float, float, float, float] = (0.28, 0.25, 1.5, 0.0, 1.0)


@dataclass(frozen=True)
class ArmConfig:
    """6-DOF elbow manipulator with a spherical wrist (analytic IK, see kinematics.py)."""

    pedestal_height: float = 0.32  # J1 frame above the floor
    shoulder_height: float = 0.20  # shoulder (J2) above the J1 frame
    upper_arm: float = 0.46
    forearm: float = 0.42  # elbow -> wrist centre
    tool_length: float = 0.36  # wrist centre -> racket face centre (along the handle)
    face_half: tuple[float, float, float] = (0.15, 0.115, 0.012)  # racket face box (string bed)
    lower_deg: tuple[float, ...] = (-160.0, -75.0, -160.0, -180.0, -125.0, -180.0)
    upper_deg: tuple[float, ...] = (160.0, 110.0, 160.0, 180.0, 125.0, 180.0)
    vmax: tuple[float, ...] = (5.0, 5.0, 6.0, 9.0, 9.0, 12.0)  # [rad/s]
    amax: tuple[float, ...] = (35.0, 35.0, 45.0, 80.0, 80.0, 120.0)  # [rad/s^2]
    masses: tuple[float, ...] = (3.0, 2.6, 1.6, 0.5, 0.4, 0.45)
    armature: tuple[float, ...] = (0.06, 0.06, 0.03, 0.01, 0.01, 0.005)
    damping: float = 0.3
    servo_bandwidth: float = 45.0  # position-servo natural frequency [rad/s]
    torque_margin: float = 3.0  # torque limit = margin * M_ii * amax


@dataclass(frozen=True)
class GantryConfig:
    root_x: float = 3.65  # |x| of the robot root (behind the baseline)
    x_range: tuple[float, float] = (0.0, 1.5)  # travel towards the net
    y_range: tuple[float, float] = (-0.9, 1.9)  # asymmetric: the forehand side is -y
    vmax: tuple[float, float] = (3.0, 3.0)
    amax: tuple[float, float] = (12.0, 16.0)
    bridge_mass: float = 10.0
    carriage_mass: float = 8.0
    servo_bandwidth: float = 30.0
    force_margin: float = 2.5


@dataclass(frozen=True)
class ContactConfig:
    court_restitution: float = 0.75
    court_friction: float = 0.6
    racket_restitution: float = 0.85
    racket_friction: float = 0.4
    net_restitution: float = 0.15
    net_friction: float = 0.6
    body_restitution: float = 0.5
    frequency: float = 300.0  # contact spring natural frequency [rad/s] (~10 ms ball contact)
    racket_frequency: float = 300.0  # softer string bed: ~10 ms contact (resolved by 10 steps)
    solimp: tuple[float, float, float] = (0.99, 0.99, 0.001)


@dataclass(frozen=True)
class PhysicsConfig:
    timestep: float = 0.001  # 1 kHz
    servo_steps: int = 2  # trajectory set-points sent to the servos at 500 Hz
    control_dt: float = 0.02  # decision period of the players (50 Hz)
    air_density: float = 1.2
    gravity: float = 9.81  # [m/s^2] (planet worlds: tennis.vla.worlds)
    wind: tuple[float, float, float] = (0.0, 0.0, 0.0)


@dataclass(frozen=True)
class TennisConfig:
    court: CourtConfig = field(default_factory=CourtConfig)
    ball: BallConfig = field(default_factory=BallConfig)
    arm: ArmConfig = field(default_factory=ArmConfig)
    gantry: GantryConfig = field(default_factory=GantryConfig)
    contact: ContactConfig = field(default_factory=ContactConfig)
    physics: PhysicsConfig = field(default_factory=PhysicsConfig)

    @property
    def frame_skip(self) -> int:
        return max(1, round(self.physics.control_dt / self.physics.timestep))
