"""Configuration dataclasses for the physical simulation (SI units: m, kg, s)."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class TableConfig:
    length: float = 2.0  # playing surface along x (goal to goal)
    width: float = 1.0  # playing surface along y
    goal_width: float = 0.30
    wall_thickness: float = 0.05
    wall_height: float = 0.04
    goal_depth: float = 0.15  # pocket depth behind the goal line


@dataclass(frozen=True)
class PuckConfig:
    radius: float = 0.032
    mass: float = 0.025
    # Air-cushion drag modelled as isotropic viscous damping: dv/dt = -decay * v.
    decay: float = 0.15  # [1/s]
    thickness: float = 0.008  # visual only


@dataclass(frozen=True)
class MalletConfig:
    radius: float = 0.048


@dataclass(frozen=True)
class RobotConfig:
    """2-DOF Cartesian gantry: X joint moves the bridge (+carriage), Y joint the carriage."""

    bridge_mass: float = 1.2
    carriage_mass: float = 0.4
    vmax: tuple[float, float] = (3.0, 3.0)  # per-axis velocity limit [m/s]
    amax: tuple[float, float] = (15.0, 25.0)  # per-axis acceleration limit [m/s^2]
    servo_time_constant: float = 0.008  # velocity-servo bandwidth (kv = mass / tau)
    force_margin: float = 1.15  # actuator force limit = margin * mass * amax
    armature: float = 0.02  # reflected motor inertia [kg]
    joint_damping: float = 0.5  # [N s/m]
    position_gain: float = 40.0  # motion-controller proportional gain near the target [1/s]
    brake_ratio: float = 0.85  # fraction of amax planned for braking
    home_offset: float = 0.16  # home x distance from own goal line


@dataclass(frozen=True)
class ContactConfig:
    """Soft-contact model: a spring-damper with natural frequency ``frequency`` [rad/s].

    MuJoCo ``solref = (1 / (frequency * zeta), zeta)`` with ``zeta`` chosen for the
    requested coefficient of restitution (see ``mjcf.contact_solref``). With a 1 kHz
    physics step this reproduces the target restitution within about +-0.03
    (tests/physics/test_physics.py).
    """

    wall_restitution: float = 0.85
    mallet_restitution: float = 0.70
    frequency: float = 600.0
    solimp: tuple[float, float, float] = (0.99, 0.99, 0.001)


@dataclass(frozen=True)
class PhysicsConfig:
    timestep: float = 0.001  # MuJoCo step (1 kHz; velocity servos act every step)
    control_dt: float = 0.02  # high-level decision period (50 Hz)
    motion_steps: int = 4  # physics steps per motion-controller update (250 Hz)


@dataclass(frozen=True)
class SimConfig:
    table: TableConfig = field(default_factory=TableConfig)
    puck: PuckConfig = field(default_factory=PuckConfig)
    mallet: MalletConfig = field(default_factory=MalletConfig)
    robot: RobotConfig = field(default_factory=RobotConfig)
    opponent_robot: RobotConfig = field(default_factory=RobotConfig)
    contact: ContactConfig = field(default_factory=ContactConfig)
    physics: PhysicsConfig = field(default_factory=PhysicsConfig)

    @property
    def frame_skip(self) -> int:
        return max(1, round(self.physics.control_dt / self.physics.timestep))
