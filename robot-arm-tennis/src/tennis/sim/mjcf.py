"""Procedural MJCF for the robot-arm tennis world.

Both robots are generated from the same template in their *root frame* (facing +x);
the away robot's root is rotated 180 deg about z. Joint chain per robot:

    {p}_jx (slide x) -> {p}_jy (slide y) -> {p}_j1 yaw (z) -> {p}_j2 shoulder pitch (-y)
    -> {p}_j3 elbow pitch (-y) -> {p}_j4 forearm roll (x) -> {p}_j5 wrist pitch (-y)
    -> {p}_j6 tool roll (x) -> racket (face centre site {p}_racket: x = handle, z = face normal)

All bodies are at their parents' origins at q = 0 except the documented link offsets, so
the analytic kinematics in ``kinematics.py`` match MuJoCo exactly (tested).
Only explicit ``<pair>`` contacts exist (ball-floor, ball-net, ball-racket, ball-robot body).
"""

from __future__ import annotations

import math

import numpy as np

from tennis.sim.config import ArmConfig, GantryConfig, TennisConfig

PLAYERS = ("home", "away")
ARM_JOINTS = ("j1", "j2", "j3", "j4", "j5", "j6")
JOINTS = ("jx", "jy", *ARM_JOINTS)
COLORS = {
    "home": ("0.12 0.42 0.95 1", "0.55 0.75 1 1"),
    "away": ("0.92 0.20 0.16 1", "1 0.6 0.55 1"),
}


def dampratio_for_restitution(e: float) -> float:
    e = float(np.clip(e, 1e-3, 0.999))
    ln_e = math.log(e)
    return -ln_e / math.sqrt(math.pi**2 + ln_e**2)


def contact_solref(restitution: float, frequency: float) -> tuple[float, float]:
    """Return (timeconst, dampratio) while keeping the contact spring frequency fixed."""
    zeta = dampratio_for_restitution(restitution)
    return 1.0 / (frequency * zeta), zeta


def _f(v) -> str:
    return " ".join(f"{float(x):.6g}" for x in np.atleast_1d(v))


def _xyaxes(pos, target) -> str:
    pos, target = np.asarray(pos, float), np.asarray(target, float)
    fwd = (target - pos) / np.linalg.norm(target - pos)
    right = np.cross(fwd, [0.0, 0.0, 1.0])
    right /= np.linalg.norm(right)
    return _f([*right, *np.cross(right, fwd)])


def _rod_inertia(m: float, length: float, radius: float) -> str:
    ia = 0.5 * m * radius**2
    it = m * (3 * radius**2 + length**2) / 12.0
    return _f([ia, it, it])


def robot_xml(p: str, arm: ArmConfig, g: GantryConfig) -> str:
    lo, hi = np.radians(arm.lower_deg), np.radians(arm.upper_deg)
    h0, d1, a2, a3, dt = (
        arm.pedestal_height,
        arm.shoulder_height,
        arm.upper_arm,
        arm.forearm,
        arm.tool_length,
    )
    fx, fy, fz = arm.face_half
    m1, m2, m3, m4, m5, m6 = arm.masses
    ylo, yhi = g.y_range
    rail_y0, rail_y1 = ylo - 0.25, yhi + 0.25
    beam_len = (rail_y1 - rail_y0) / 2
    beam_mid = (rail_y1 + rail_y0) / 2
    xmid = (g.x_range[0] + g.x_range[1]) / 2
    xhalf = (g.x_range[1] - g.x_range[0]) / 2 + 0.25
    j = lambda name, axis, k: (  # noqa: E731
        f'<joint name="{p}_{name}" type="hinge" axis="{axis}" range="{lo[k]:.5f} {hi[k]:.5f}" limited="true" '
        f'armature="{arm.armature[k]}" damping="{arm.damping}"/>'
    )
    return f"""
    <geom name="{p}_rail_a" type="box" pos="{xmid:.4f} {rail_y0:.4f} 0.02" size="{xhalf:.4f} 0.03 0.02" material="gantry_dark"/>
    <geom name="{p}_rail_b" type="box" pos="{xmid:.4f} {rail_y1:.4f} 0.02" size="{xhalf:.4f} 0.03 0.02" material="gantry_dark"/>
    <body name="{p}_gx">
      <joint name="{p}_jx" type="slide" axis="1 0 0" range="{g.x_range[0]} {g.x_range[1]}" limited="true" armature="0.5" damping="5"/>
      <inertial pos="0 {beam_mid:.4f} 0.07" mass="{g.bridge_mass}" diaginertia="5 0.2 5"/>
      <geom name="{p}_bridge" type="box" pos="0 {beam_mid:.4f} 0.07" size="0.05 {beam_len:.4f} 0.035" material="gantry"/>
      <body name="{p}_gy">
        <joint name="{p}_jy" type="slide" axis="0 1 0" range="{ylo} {yhi}" limited="true" armature="0.5" damping="5"/>
        <inertial pos="0 0 0.15" mass="{g.carriage_mass}" diaginertia="0.2 0.2 0.1"/>
        <geom name="{p}_carriage" type="box" pos="0 0 0.12" size="0.16 0.14 0.02" material="{p}_accent"/>
        <geom name="{p}_pedestal" type="cylinder" pos="0 0 {(0.14 + h0) / 2:.4f}" size="0.075 {(h0 - 0.14) / 2:.4f}" material="arm_dark"/>
        <body name="{p}_link1" pos="0 0 {h0}" gravcomp="1">
          {j("j1", "0 0 1", 0)}
          <inertial pos="0 0 0.1" mass="{m1}" diaginertia="0.02 0.02 0.015"/>
          <geom name="{p}_link1_col" type="cylinder" pos="0 0 0.09" size="0.085 0.09" material="arm_white"/>
          <geom type="cylinder" pos="0 0 0.012" size="0.09 0.012" material="{p}_accent"/>
          <body name="{p}_upper" pos="0 0 {d1}" gravcomp="1">
            {j("j2", "0 -1 0", 1)}
            <inertial pos="{a2 / 2:.4f} 0 0" mass="{m2}" diaginertia="{_rod_inertia(m2, a2, 0.05)}"/>
            <geom type="cylinder" zaxis="0 1 0" size="0.072 0.07" material="arm_dark"/>
            <geom type="cylinder" zaxis="0 1 0" size="0.074 0.015" pos="0 0.072 0" material="{p}_accent"/>
            <geom name="{p}_upper_col" type="capsule" fromto="0.05 0 0 {a2 - 0.03:.4f} 0 0" size="0.05" material="arm_white"/>
            <body name="{p}_fore" pos="{a2} 0 0" gravcomp="1">
              {j("j3", "0 -1 0", 2)}
              <inertial pos="{a3 / 2:.4f} 0 0" mass="{m3}" diaginertia="{_rod_inertia(m3, a3, 0.04)}"/>
              <geom type="cylinder" zaxis="0 1 0" size="0.058 0.055" material="arm_dark"/>
              <geom name="{p}_fore_col" type="capsule" fromto="0.04 0 0 {a3 - 0.05:.4f} 0 0" size="0.04" material="arm_white"/>
              <body name="{p}_w1" pos="{a3} 0 0" gravcomp="1">
                {j("j4", "1 0 0", 3)}
                <inertial pos="-0.03 0 0" mass="{m4}" diaginertia="0.0006 0.0006 0.0006"/>
                <geom type="cylinder" zaxis="1 0 0" pos="-0.04 0 0" size="0.042 0.03" material="{p}_accent"/>
                <body name="{p}_w2" gravcomp="1">
                  {j("j5", "0 -1 0", 4)}
                  <inertial pos="0 0 0" mass="{m5}" diaginertia="0.0004 0.0004 0.0004"/>
                  <geom type="sphere" size="0.04" material="arm_dark"/>
                  <body name="{p}_w3" gravcomp="1">
                    {j("j6", "1 0 0", 5)}
                    <inertial pos="{dt * 0.55:.4f} 0 0" mass="{m6}" diaginertia="0.0002 0.006 0.006"/>
                    <geom type="cylinder" zaxis="1 0 0" pos="0.035 0 0" size="0.032 0.018" material="arm_white"/>
                    <geom type="capsule" fromto="0.06 0 0 {dt - fx - 0.02:.4f} 0 0" size="0.015" material="grip"/>
                    <geom name="{p}_racket_frame" type="ellipsoid" pos="{dt} 0 0" size="{fx + 0.014:.4f} {fy + 0.014:.4f} 0.009" material="{p}_racket"/>
                    <geom name="{p}_racket_strings" type="ellipsoid" pos="{dt} 0 0" size="{fx:.4f} {fy:.4f} 0.0105" material="strings"/>
                    <geom name="{p}_racket_col" type="box" pos="{dt} 0 0" size="{fx} {fy} {fz}" rgba="0 0 0 0" group="3"/>
                    <site name="{p}_racket" pos="{dt} 0 0" size="0.01" rgba="1 1 1 0"/>
                  </body>
                </body>
              </body>
            </body>
          </body>
        </body>
      </body>
    </body>"""


def body_collision_geoms(p: str) -> tuple[str, ...]:
    return (f"{p}_link1_col", f"{p}_upper_col", f"{p}_fore_col", f"{p}_pedestal")


def build_mjcf(cfg: TennisConfig) -> str:
    c, b, k, ph = cfg.court, cfg.ball, cfg.contact, cfg.physics
    L2, W2, lw = c.length / 2, c.width / 2, c.line_width
    rx = cfg.gantry.root_x
    solimp = _f(k.solimp)
    ref = lambda e: _f(contact_solref(e, k.frequency))  # noqa: E731
    fric_ref = _f((0.004, 1.0))
    pairs = [
        f'<pair name="ball_floor" geom1="ball" geom2="floor" condim="3" friction="{k.court_friction} {k.court_friction} 0.005 0.0001 0.0001" solref="{ref(k.court_restitution)}" solreffriction="{fric_ref}" solimp="{solimp}"/>',
        f'<pair name="ball_net" geom1="ball" geom2="net" condim="3" friction="{k.net_friction} {k.net_friction} 0.005 0.0001 0.0001" solref="{ref(k.net_restitution)}" solreffriction="{fric_ref}" solimp="{solimp}"/>',
    ]
    for p in PLAYERS:
        racket_ref = _f(contact_solref(k.racket_restitution, k.racket_frequency))
        pairs.append(
            f'<pair name="ball_{p}_racket" geom1="ball" geom2="{p}_racket_col" condim="3" friction="{k.racket_friction} {k.racket_friction} 0.005 0.0001 0.0001" solref="{racket_ref}" solreffriction="{fric_ref}" solimp="{solimp}"/>'
        )
        for gname in body_collision_geoms(p):
            pairs.append(
                f'<pair name="ball_{gname}" geom1="ball" geom2="{gname}" condim="1" solref="{ref(k.body_restitution)}" solimp="{solimp}"/>'
            )
    lines = [
        ("base_home", (-L2, 0), (lw / 2, W2 + lw / 2)),
        ("base_away", (L2, 0), (lw / 2, W2 + lw / 2)),
        ("side_pos", (0, W2), (L2 + lw / 2, lw / 2)),
        ("side_neg", (0, -W2), (L2 + lw / 2, lw / 2)),
        ("service_home", (-L2 * 0.55, 0), (lw / 2, W2)),
        ("service_away", (L2 * 0.55, 0), (lw / 2, W2)),
        ("centre", (0, 0), (L2 * 0.55, lw / 2)),
        ("mark_home", (-L2 + 0.06, 0), (0.06, lw / 2)),
        ("mark_away", (L2 - 0.06, 0), (0.06, lw / 2)),
    ]
    lines_xml = "\n    ".join(
        f'<geom name="line_{n}" type="box" pos="{x:.4f} {y:.4f} 0.0008" size="{sx:.4f} {sy:.4f} 0.0006" material="line"/>'
        for n, (x, y), (sx, sy) in lines
    )
    cams = {
        "broadcast": ((0.0, -7.4, 3.6), (0.0, 0.0, 0.25), 40),
        "behind_home": ((-7.6, 0.0, 2.5), (0.0, 0.0, 0.45), 42),
        "top": ((0.0, 0.0, 11.0), (0.0, 0.001, 0.0), 45),
        "home_closeup": ((-1.55, -2.9, 1.25), (-3.0, -0.25, 0.65), 50),
        "side_low": ((0.0, -4.2, 0.9), (0.0, 0.0, 0.5), 60),
        "vla_front": ((-5.6, 0.0, 3.0), (-0.46, 0.0, -0.09), 64),  # VLA policy camera
    }
    cams_xml = "\n    ".join(
        f'<camera name="{n}" pos="{_f(pos)}" xyaxes="{_xyaxes(pos, tgt)}" fovy="{fov}"/>'
        for n, (pos, tgt, fov) in cams.items()
    )
    ball_inertia = 2.0 / 3.0 * b.mass * b.radius**2  # thin-walled sphere
    span, nh = c.net_half_span, c.net_height
    return f"""<mujoco model="robot_arm_tennis">
  <compiler angle="radian" autolimits="true"/>
  <option timestep="{ph.timestep}" integrator="implicitfast" gravity="0 0 -{ph.gravity}" density="{ph.air_density}" viscosity="0" wind="{_f(ph.wind)}"/>
  <visual>
    <global offwidth="1920" offheight="1080"/>
    <quality shadowsize="4096" offsamples="4"/>
    <headlight ambient="0.32 0.32 0.35" diffuse="0.4 0.4 0.4" specular="0.1 0.1 0.1"/>
    <rgba haze="0.55 0.68 0.85 1"/>
  </visual>
  <default>
    <geom contype="0" conaffinity="0"/>
  </default>
  <asset>
    <texture name="sky" type="skybox" builtin="gradient" rgb1="0.62 0.78 0.95" rgb2="0.30 0.45 0.70" width="512" height="512"/>
    <texture name="surround" type="2d" builtin="checker" rgb1="0.22 0.47 0.34" rgb2="0.21 0.45 0.33" width="512" height="512"/>
    <material name="surround" texture="surround" texrepeat="10 10" reflectance="0.02"/>
    <material name="court" rgba="0.16 0.34 0.60 1" specular="0.15"/>
    <material name="line" rgba="0.96 0.96 0.96 1"/>
    <material name="net" rgba="0.08 0.08 0.10 0.72"/>
    <material name="net_tape" rgba="0.97 0.97 0.97 1"/>
    <material name="post" rgba="0.25 0.27 0.30 1" specular="0.5"/>
    <material name="ball" rgba="0.86 0.95 0.18 1" specular="0.3"/>
    <material name="ball_seam" rgba="0.97 0.97 0.92 1"/>
    <material name="gantry" rgba="0.55 0.58 0.62 1" specular="0.5"/>
    <material name="gantry_dark" rgba="0.24 0.26 0.29 1" specular="0.4"/>
    <material name="arm_white" rgba="0.93 0.94 0.95 1" specular="0.6" shininess="0.6"/>
    <material name="arm_dark" rgba="0.20 0.21 0.24 1" specular="0.5"/>
    <material name="grip" rgba="0.10 0.10 0.11 1"/>
    <material name="strings" rgba="0.92 0.92 0.88 0.55"/>
    <material name="home_accent" rgba="{COLORS["home"][0]}" specular="0.5"/>
    <material name="away_accent" rgba="{COLORS["away"][0]}" specular="0.5"/>
    <material name="home_racket" rgba="{COLORS["home"][0]}" specular="0.7"/>
    <material name="away_racket" rgba="{COLORS["away"][0]}" specular="0.7"/>
  </asset>
  <worldbody>
    <light name="sun" pos="2 -3 9" dir="-0.2 0.3 -1" diffuse="0.62 0.62 0.6" specular="0.25 0.25 0.25" castshadow="true"/>
    <light name="fill" pos="-3 4 6" dir="0.3 -0.4 -1" diffuse="0.25 0.25 0.28" castshadow="false"/>
    {cams_xml}
    <geom name="floor" type="plane" size="9 6 0.1" material="surround"/>
    <geom name="court" type="box" pos="0 0 0.0003" size="{L2 + 0.9:.4f} {W2 + 0.6:.4f} 0.0003" material="court"/>
    {lines_xml}
    <geom name="net" type="box" pos="0 0 {nh / 2:.4f}" size="0.006 {span} {nh / 2:.4f}" material="net"/>
    <geom name="net_tape" type="box" pos="0 0 {nh - 0.012:.4f}" size="0.009 {span} 0.012" material="net_tape"/>
    <geom name="post_pos" type="cylinder" pos="0 {span + 0.03:.4f} {(nh + 0.04) / 2:.4f}" size="0.03 {(nh + 0.04) / 2:.4f}" material="post"/>
    <geom name="post_neg" type="cylinder" pos="0 {-span - 0.03:.4f} {(nh + 0.04) / 2:.4f}" size="0.03 {(nh + 0.04) / 2:.4f}" material="post"/>
    <body name="ball" pos="0 0 1">
      <freejoint name="ball"/>
      <inertial pos="0 0 0" mass="{b.mass}" diaginertia="{ball_inertia:.6g} {ball_inertia:.6g} {ball_inertia:.6g}"/>
      <geom name="ball" type="sphere" size="{b.radius}" material="ball" fluidshape="ellipsoid" fluidcoef="{_f(b.fluidcoef)}"/>
      <geom name="ball_seam" type="ellipsoid" pos="0 0 {b.radius * 0.93:.4f}" size="{b.radius * 0.45:.4f} {b.radius * 0.12:.4f} {b.radius * 0.1:.4f}" material="ball_seam" mass="0"/>
    </body>
    <body name="home_root" pos="{-rx} 0 0">
      {robot_xml("home", cfg.arm, cfg.gantry)}
    </body>
    <body name="away_root" pos="{rx} 0 0" quat="0 0 0 1">
      {robot_xml("away", cfg.arm, cfg.gantry)}
    </body>
  </worldbody>
  <contact>
    {chr(10).join("    " + q for q in pairs).strip()}
  </contact>
  <actuator>
    {chr(10).join(f'    <general name="{p}_a_{jn}" joint="{p}_{jn}" gaintype="fixed" biastype="affine" gainprm="1" biasprm="0 -1 -0.1" forcelimited="true" forcerange="-1 1"/>' for p in PLAYERS for jn in JOINTS).strip()}
  </actuator>
</mujoco>
"""
