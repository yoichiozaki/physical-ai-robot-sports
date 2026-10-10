"""Programmatic MJCF generation for the air-hockey table, puck and two gantry robots.

World frame: origin at the table centre, x along the table (home goal at x = -L/2,
away goal at x = +L/2), y across the table, z up. The playing surface is z = 0.

Collision design (see DECISIONS.md D-004/D-005): the puck and mallets are constrained
to the plane by slide joints and collide through invisible spheres whose centres share
the same height, so every contact normal is horizontal. Only explicit ``<pair>`` contacts
exist (puck-wall, puck-mallet) with ``condim=1`` (frictionless reflections).
"""

from __future__ import annotations

import math

import numpy as np

from hockey.simulation.config import RobotConfig, SimConfig

COLLISION_Z = 0.015  # height of collision sphere centres
PLAYERS = ("home", "away")
GOAL_POCKET_DAMPRATIO = 1.0
GANTRY_Z = 0.16


def dampratio_for_restitution(e: float) -> float:
    """Damping ratio of a linear spring-damper contact giving restitution ``e``.

    MuJoCo's soft contacts are close to (but not exactly) a linear spring-damper,
    so the realised restitution is verified/identified empirically (tests/physics).
    """
    e = float(np.clip(e, 1e-3, 0.999))
    ln_e = math.log(e)
    return -ln_e / math.sqrt(math.pi**2 + ln_e**2)


def contact_solref(restitution: float, frequency: float) -> tuple[float, float]:
    """MuJoCo (timeconst, dampratio) for a contact spring of natural frequency ``frequency``.

    In MuJoCo's parameterisation the stiffness is 1 / (timeconst * dampratio)^2, so the
    time constant must shrink as the damping ratio grows to keep the frequency fixed.
    """
    zeta = dampratio_for_restitution(restitution)
    return 1.0 / (frequency * zeta), zeta


def _fmt(values: tuple[float, ...] | list[float] | np.ndarray) -> str:
    return " ".join(f"{float(v):.6g}" for v in values)


def _lookat_xyaxes(pos: np.ndarray, target: np.ndarray) -> str:
    fwd = target - pos
    fwd = fwd / np.linalg.norm(fwd)
    right = np.cross(fwd, np.array([0.0, 0.0, 1.0]))
    right = right / np.linalg.norm(right)
    up = np.cross(right, fwd)
    return _fmt([*right, *up])


def wall_geoms(cfg: SimConfig) -> list[tuple[str, tuple[float, ...], tuple[float, ...], bool]]:
    """(name, pos, half-size, is_goal_back) for every static collision box."""
    t = cfg.table
    L2, W2, th, h2 = t.length / 2, t.width / 2, t.wall_thickness, t.wall_height / 2
    g2, depth = t.goal_width / 2, t.goal_depth
    seg = (W2 - g2) / 2
    walls: list[tuple[str, tuple[float, ...], tuple[float, ...], bool]] = [
        ("wall_side_pos", (0.0, W2 + th / 2, h2), (L2 + th, th / 2, h2), False),
        ("wall_side_neg", (0.0, -W2 - th / 2, h2), (L2 + th, th / 2, h2), False),
    ]
    for side, sx in (("home", -1.0), ("away", 1.0)):
        for tag, sy in (("pos", 1.0), ("neg", -1.0)):
            walls.append(
                (
                    f"wall_end_{side}_{tag}",
                    (sx * (L2 + th / 2), sy * (g2 + seg), h2),
                    (th / 2, seg, h2),
                    False,
                )
            )
            walls.append(
                (
                    f"goal_{side}_side_{tag}",
                    (sx * (L2 + depth / 2 + th / 2), sy * (g2 + th / 2), h2),
                    (depth / 2 + th / 2, th / 2, h2),
                    False,
                )
            )
        walls.append(
            (
                f"goal_{side}_back",
                (sx * (L2 + depth + th / 2), 0.0, h2),
                (th / 2, g2 + th, h2),
                True,
            )
        )
    return walls


def joint_ranges(cfg: SimConfig, player: str) -> tuple[tuple[float, float], tuple[float, float]]:
    """World-frame joint ranges of a player's mallet centre (x, y)."""
    L2, W2, rm = cfg.table.length / 2, cfg.table.width / 2, cfg.mallet.radius
    xr = (-L2 + rm, -rm) if player == "home" else (rm, L2 - rm)
    yr = (-W2 + rm, W2 - rm)
    return xr, yr


def _robot_xml(cfg: SimConfig, player: str, robot: RobotConfig, color: str) -> str:
    t = cfg.table
    sx = -1.0 if player == "home" else 1.0
    back = 0.075 * sx  # bridge sits behind the mallet (towards own goal)
    span = t.width / 2 + t.wall_thickness + 0.07
    (xlo, xhi), (ylo, yhi) = joint_ranges(cfg, player)
    m_c = robot.carriage_mass
    m_b = robot.bridge_mass
    rm = cfg.mallet.radius
    return f"""
    <body name="{player}_bridge" pos="0 0 0">
      <joint name="{player}_x" type="slide" axis="1 0 0" range="{xlo:.6g} {xhi:.6g}" limited="true"
             armature="{robot.armature}" damping="{robot.joint_damping}" solreflimit="0.004 1"/>
      <inertial pos="{back:.6g} 0 {GANTRY_Z}" mass="{m_b}" diaginertia="0.05 0.002 0.05"/>
      <geom name="{player}_bridge_beam" type="box" pos="{back:.6g} 0 {GANTRY_Z}" size="0.022 {span:.6g} 0.014"
            material="gantry" group="1"/>
      <geom name="{player}_bridge_end_pos" type="box" pos="{back:.6g} {span:.6g} {GANTRY_Z - 0.02}"
            size="0.035 0.03 0.03" material="gantry_dark" group="1"/>
      <geom name="{player}_bridge_end_neg" type="box" pos="{back:.6g} {-span:.6g} {GANTRY_Z - 0.02}"
            size="0.035 0.03 0.03" material="gantry_dark" group="1"/>
      <body name="{player}_carriage" pos="0 0 0">
        <joint name="{player}_y" type="slide" axis="0 1 0" range="{ylo:.6g} {yhi:.6g}" limited="true"
               armature="{robot.armature}" damping="{robot.joint_damping}" solreflimit="0.004 1"/>
        <inertial pos="0 0 0.03" mass="{m_c}" diaginertia="0.001 0.001 0.001"/>
        <geom name="{player}_carriage_block" type="box" pos="{back:.6g} 0 {GANTRY_Z + 0.022}"
              size="0.04 0.045 0.012" material="gantry_dark" group="1"/>
        <geom name="{player}_arm" type="capsule" fromto="{back:.6g} 0 {GANTRY_Z} 0 0 0.075" size="0.009"
              material="gantry_dark" group="1"/>
        <geom name="{player}_mallet_col" type="sphere" size="{rm}" pos="0 0 {COLLISION_Z}"
              rgba="0 0 0 0" group="3"/>
        <geom name="{player}_mallet_base" type="cylinder" size="{rm} 0.011" pos="0 0 0.011"
              material="{color}" group="1"/>
        <geom name="{player}_mallet_rim" type="cylinder" size="{rm * 0.72:.6g} 0.004" pos="0 0 0.024"
              material="{color}_light" group="1"/>
        <geom name="{player}_mallet_handle" type="cylinder" size="0.018 0.022" pos="0 0 0.048"
              material="{color}" group="1"/>
        <geom name="{player}_mallet_knob" type="sphere" size="0.024" pos="0 0 0.074"
              material="{color}" group="1"/>
      </body>
    </body>"""


def _actuator_xml(player: str, robot: RobotConfig) -> str:
    m_x = robot.bridge_mass + robot.carriage_mass + robot.armature
    m_y = robot.carriage_mass + robot.armature
    lines = []
    for axis, mass, vmax, amax in (
        ("x", m_x, robot.vmax[0], robot.amax[0]),
        ("y", m_y, robot.vmax[1], robot.amax[1]),
    ):
        kv = mass / robot.servo_time_constant
        fmax = robot.force_margin * mass * amax
        lines.append(
            f'<velocity name="{player}_{axis}_servo" joint="{player}_{axis}" kv="{kv:.6g}" '
            f'ctrlrange="{-vmax:.6g} {vmax:.6g}" ctrllimited="true" '
            f'forcerange="{-fmax:.6g} {fmax:.6g}" forcelimited="true"/>'
        )
    return "\n    ".join(lines)


def build_mjcf(cfg: SimConfig) -> str:
    t, p, c = cfg.table, cfg.puck, cfg.contact
    L2, W2, th = t.length / 2, t.width / 2, t.wall_thickness
    dt = cfg.physics.timestep
    puck_damping = p.decay * p.mass

    walls_xml = []
    for name, pos, size, _is_back in wall_geoms(cfg):
        mat = "goal_wall" if name.startswith("goal_") else "wall"
        walls_xml.append(
            f'<geom name="{name}" type="box" pos="{_fmt(pos)}" size="{_fmt(size)}" material="{mat}"/>'
        )

    solimp = _fmt(c.solimp)
    ref_wall = _fmt(contact_solref(c.wall_restitution, c.frequency))
    ref_mallet = _fmt(contact_solref(c.mallet_restitution, c.frequency))
    ref_pocket = _fmt((1.0 / (c.frequency * GOAL_POCKET_DAMPRATIO), GOAL_POCKET_DAMPRATIO))
    pairs = []
    for name, _pos, _size, _is_back in wall_geoms(cfg):
        ref = ref_pocket if name.startswith("goal_") else ref_wall
        pairs.append(
            f'<pair name="puck_{name}" geom1="puck_col" geom2="{name}" condim="1" '
            f'solref="{ref}" solimp="{solimp}"/>'
        )
    for player in PLAYERS:
        pairs.append(
            f'<pair name="puck_{player}_mallet" geom1="puck_col" geom2="{player}_mallet_col" condim="1" '
            f'solref="{ref_mallet}" solimp="{solimp}"/>'
        )

    # Table markings (visual only, slightly above the surface).
    g2 = t.goal_width / 2
    markings = [
        f'<geom name="mark_center" type="box" pos="0 0 0.0006" size="0.006 {W2:.6g} 0.0005" rgba="0.85 0.15 0.15 1"/>',
        '<geom name="mark_circle" type="cylinder" pos="0 0 0.0004" size="0.13 0.0003" rgba="0.85 0.15 0.15 0.35"/>',
        '<geom name="mark_circle_in" type="cylinder" pos="0 0 0.0005" size="0.122 0.0003" rgba="0.93 0.95 0.98 1"/>',
    ]
    for side, sx in (("home", -1.0), ("away", 1.0)):
        markings.append(
            f'<geom name="mark_crease_{side}" type="cylinder" pos="{sx * L2:.6g} 0 0.0004" '
            f'size="{g2 + 0.035:.6g} 0.0003" rgba="0.2 0.35 0.8 0.22"/>'
        )
        markings.append(
            f'<geom name="mark_blue_{side}" type="box" pos="{sx * L2 * 0.5:.6g} 0 0.0006" '
            f'size="0.004 {W2:.6g} 0.0005" rgba="0.2 0.35 0.8 0.8"/>'
        )

    rail_y = W2 + th + 0.07
    rails = []
    for side, sx in (("home", -1.0), ("away", 1.0)):
        for tag, sy in (("pos", 1.0), ("neg", -1.0)):
            rails.append(
                f'<geom name="rail_{side}_{tag}" type="box" pos="{sx * (L2 + th) / 2:.6g} {sy * rail_y:.6g} '
                f'{GANTRY_Z - 0.045:.6g}" size="{(L2 + th) / 2:.6g} 0.018 0.012" material="gantry_dark"/>'
            )
            rails.append(
                f'<geom name="post_{side}_{tag}" type="box" pos="{sx * (L2 + th - 0.02):.6g} {sy * rail_y:.6g} '
                f'{(GANTRY_Z - 0.06) / 2:.6g}" size="0.02 0.02 {(GANTRY_Z - 0.06) / 2:.6g}" material="gantry_dark"/>'
            )

    # Cameras
    view_w = t.length + 2 * th + 0.25
    top_h = 2.6
    fovy_top = math.degrees(2 * math.atan((view_w * 9 / 16) / 2 / top_h))
    bc_pos = np.array([0.0, -1.62, 1.5])
    bc_target = np.array([0.0, 0.1, -0.06])
    hv_pos = np.array([-L2 - 0.95, -0.85, 0.95])
    hv_target = np.array([-0.15, 0.0, 0.0])

    return f"""<mujoco model="adaptive_air_hockey">
  <compiler angle="radian" autolimits="true"/>
  <option timestep="{dt}" integrator="implicitfast" gravity="0 0 -9.81"/>
  <visual>
    <global offwidth="1920" offheight="1080"/>
    <quality shadowsize="4096" offsamples="4"/>
    <headlight ambient="0.35 0.35 0.38" diffuse="0.45 0.45 0.45" specular="0.1 0.1 0.1"/>
    <rgba haze="0.15 0.18 0.24 1"/>
  </visual>
  <default>
    <geom contype="0" conaffinity="0"/>
  </default>
  <asset>
    <texture name="sky" type="skybox" builtin="gradient" rgb1="0.20 0.24 0.32" rgb2="0.05 0.06 0.09" width="512" height="512"/>
    <texture name="floor" type="2d" builtin="checker" rgb1="0.16 0.17 0.20" rgb2="0.20 0.21 0.24" width="512" height="512"/>
    <material name="floor" texture="floor" texrepeat="8 8" reflectance="0.05"/>
    <material name="table" rgba="0.93 0.95 0.98 1" reflectance="0.08" specular="0.3"/>
    <material name="table_frame" rgba="0.13 0.16 0.22 1"/>
    <material name="wall" rgba="0.16 0.22 0.34 1" specular="0.4"/>
    <material name="goal_wall" rgba="0.10 0.12 0.16 1"/>
    <material name="puck" rgba="0.08 0.08 0.10 1" specular="0.6" shininess="0.6"/>
    <material name="puck_top" rgba="0.98 0.78 0.10 1"/>
    <material name="blue" rgba="0.10 0.38 0.92 1" specular="0.5" shininess="0.5"/>
    <material name="blue_light" rgba="0.45 0.65 1.0 1"/>
    <material name="red" rgba="0.90 0.16 0.14 1" specular="0.5" shininess="0.5"/>
    <material name="red_light" rgba="1.0 0.55 0.50 1"/>
    <material name="gantry" rgba="0.70 0.73 0.78 0.55" specular="0.6"/>
    <material name="gantry_dark" rgba="0.30 0.32 0.36 1" specular="0.4"/>
  </asset>
  <worldbody>
    <light name="key" pos="0 -1.0 3.0" dir="0 0.3 -1" diffuse="0.55 0.55 0.55" specular="0.2 0.2 0.2" castshadow="true"/>
    <light name="fill" pos="0 1.5 2.5" dir="0 -0.5 -1" diffuse="0.25 0.25 0.28" castshadow="false"/>
    <camera name="top" pos="0 0 {top_h}" xyaxes="1 0 0 0 1 0" fovy="{fovy_top:.4f}"/>
    <camera name="broadcast" pos="{_fmt(bc_pos)}" xyaxes="{_lookat_xyaxes(bc_pos, bc_target)}" fovy="42"/>
    <camera name="home_view" pos="{_fmt(hv_pos)}" xyaxes="{_lookat_xyaxes(hv_pos, hv_target)}" fovy="50"/>
    <geom name="floor" type="plane" pos="0 0 -0.78" size="6 6 0.1" material="floor"/>
    <geom name="table_top" type="box" pos="0 0 -0.02" size="{L2 + th:.6g} {W2 + th:.6g} 0.02" material="table"/>
    <geom name="goal_floor_home" type="box" pos="{-(L2 + th + t.goal_depth / 2):.6g} 0 -0.0185"
          size="{t.goal_depth / 2 + th:.6g} {g2 + th:.6g} 0.02" material="goal_wall"/>
    <geom name="goal_floor_away" type="box" pos="{L2 + th + t.goal_depth / 2:.6g} 0 -0.0185"
          size="{t.goal_depth / 2 + th:.6g} {g2 + th:.6g} 0.02" material="goal_wall"/>
    <geom name="table_body" type="box" pos="0 0 -0.25" size="{L2 + th + 0.02:.6g} {W2 + th + 0.02:.6g} 0.21"
          material="table_frame"/>
    <geom name="table_leg" type="box" pos="0 0 -0.6" size="{L2 * 0.6:.6g} {W2 * 0.5:.6g} 0.18" material="table_frame"/>
    {chr(10).join("    " + m for m in markings).strip()}
    {chr(10).join("    " + w for w in walls_xml).strip()}
    {chr(10).join("    " + r for r in rails).strip()}
    <body name="puck" pos="0 0 0">
      <joint name="puck_x" type="slide" axis="1 0 0" damping="{puck_damping:.6g}"/>
      <joint name="puck_y" type="slide" axis="0 1 0" damping="{puck_damping:.6g}"/>
      <inertial pos="0 0 {COLLISION_Z}" mass="{p.mass}" diaginertia="1e-5 1e-5 1e-5"/>
      <geom name="puck_col" type="sphere" size="{p.radius}" pos="0 0 {COLLISION_Z}" rgba="0 0 0 0" group="3"/>
      <geom name="puck_body" type="cylinder" size="{p.radius} {p.thickness / 2:.6g}" pos="0 0 {p.thickness / 2:.6g}"
            material="puck" group="1"/>
      <geom name="puck_top" type="cylinder" size="{p.radius * 0.55:.6g} 0.0006" pos="0 0 {p.thickness + 0.0006:.6g}"
            material="puck_top" group="1"/>
    </body>
    {_robot_xml(cfg, "home", cfg.robot, "blue")}
    {_robot_xml(cfg, "away", cfg.opponent_robot, "red")}
  </worldbody>
  <contact>
    {chr(10).join("    " + q for q in pairs).strip()}
  </contact>
  <actuator>
    {_actuator_xml("home", cfg.robot)}
    {_actuator_xml("away", cfg.opponent_robot)}
  </actuator>
</mujoco>
"""
