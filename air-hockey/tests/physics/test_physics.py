"""Physics tests (Phase 2): stability, reflections, collisions, limits, goals, reproducibility."""

from __future__ import annotations

import math

import numpy as np
import pytest

from hockey.controllers.opponents import make_opponent
from hockey.controllers.physics_based import PhysicsController
from hockey.evaluation.match import play_match
from hockey.simulation import HockeySim, PhysicsParams, SimConfig
from hockey.simulation.game import GameConfig


def _wall_restitution(sim: HockeySim, speed: float, angle_deg: float) -> tuple[float, float]:
    a = math.radians(angle_deg)
    v0 = np.array([speed * math.cos(a), speed * math.sin(a)])
    sim.reset(puck_pos=(0.3, 0.2), puck_vel=v0)
    vin = None
    vt_in = None
    for _ in range(200):
        r = sim.step()
        for e in r.events:
            if e.kind == "wall" and vin is None:
                vin, vt_in = e.puck_vel_before[1], e.puck_vel_before[0]
        if vin is not None and not sim._in_contact["wall"]:
            st = sim.state()
            return -st.puck_vel[1] / vin, (st.puck_vel[0] / vt_in if abs(vt_in) > 1e-9 else 1.0)
    raise AssertionError("puck never bounced")


def test_stationary_puck_is_stable(sim: HockeySim) -> None:
    sim.reset(puck_pos=(0.2, -0.1), puck_vel=(0.0, 0.0))
    for _ in range(100):  # 2 s
        sim.step()
    st = sim.state()
    assert np.allclose(st.puck_pos, (0.2, -0.1), atol=1e-6)
    assert np.allclose(st.puck_vel, 0.0, atol=1e-6)
    # robots hold their home positions
    assert np.allclose(st.home_pos, sim.home_position("home"), atol=1e-3)


@pytest.mark.parametrize("target_e", [0.75, 0.85, 0.95])
def test_wall_reflection_matches_restitution(target_e: float) -> None:
    sim = HockeySim()
    sim.set_params(
        PhysicsParams(puck_decay=0.15, wall_restitution=target_e, mallet_restitution=0.7)
    )
    for speed in (0.5, 2.0, 5.0):
        for angle in (90.0, 40.0):
            e, tangential = _wall_restitution(sim, speed, angle)
            assert e == pytest.approx(target_e, abs=0.04), (speed, angle, e)
            assert tangential == pytest.approx(1.0, abs=0.02)  # frictionless contact


def test_mallet_collision_bounces_puck(sim: HockeySim) -> None:
    sim.reset(puck_pos=(-0.3, 0.0), puck_vel=(-3.0, 0.0), home_pos=(-0.8, 0.0))
    events = []
    for _ in range(100):
        r = sim.step()
        events += [e.kind for e in r.events]
        if "home" in events and sim.state().puck_vel[0] > 0 and not sim._in_contact["home"]:
            break
    st = sim.state()
    assert "home" in events
    assert st.puck_vel[0] > 0.0
    # identified effective restitution of a servo-held mallet ~0.68 (see EXPERIMENTS.md)
    assert st.puck_vel[0] / 3.0 == pytest.approx(0.68, abs=0.06)


def test_no_tunnelling_at_high_speed(sim: HockeySim, rng: np.random.Generator) -> None:
    cfg = sim.cfg
    L2, W2, rp = cfg.table.length / 2, cfg.table.width / 2, cfg.puck.radius
    max_pen = 0.0
    for _ in range(10):
        ang = rng.uniform(-math.pi, math.pi)
        sim.reset(
            puck_pos=(rng.uniform(-0.5, 0.5), rng.uniform(-0.3, 0.3)),
            puck_vel=(8 * math.cos(ang), 8 * math.sin(ang)),
        )
        for _ in range(100):
            r = sim.step()
            assert not r.out_of_bounds
            st = sim.state()
            max_pen = max(max_pen, abs(st.puck_pos[1]) + rp - W2)
            if r.goal:
                break
            assert abs(st.puck_pos[0]) < L2 + cfg.table.goal_depth
    assert max_pen < 0.015  # soft-contact penetration below 1.5 cm at 8 m/s


def test_robot_respects_joint_and_motion_limits(sim: HockeySim) -> None:
    rc = sim.cfg.robot
    lx, ly = sim.limits["home"]
    (xlo, xhi), (ylo, yhi) = (lx.qmin, lx.qmax), (ly.qmin, ly.qmax)
    targets = [(-2.0, 2.0), (1.0, -2.0), (-0.1, 0.0), (-0.9, 0.45), (0.5, 0.5)]
    prev_v = np.zeros(2)
    for tx, ty in targets:
        sim.set_target("home", (tx, ty))
        for _ in range(40):
            sim.step()
            st = sim.state()
            assert xlo - 2e-3 <= st.home_pos[0] <= xhi + 2e-3
            assert ylo - 2e-3 <= st.home_pos[1] <= yhi + 2e-3
            assert st.home_pos[0] < 0.0  # never crosses the centre line
            assert abs(st.home_vel[0]) <= rc.vmax[0] * 1.05
            assert abs(st.home_vel[1]) <= rc.vmax[1] * 1.05
            acc = (st.home_vel - prev_v) / sim.control_dt
            assert abs(acc[0]) <= rc.amax[0] * rc.force_margin * 1.1
            assert abs(acc[1]) <= rc.amax[1] * rc.force_margin * 1.1
            prev_v = st.home_vel


def test_numerical_stability_random_play(sim: HockeySim, rng: np.random.Generator) -> None:
    sim.reset(puck_pos=(0.0, 0.0), puck_vel=(2.0, 1.0))
    for i in range(1500):  # 30 s
        if i % 10 == 0:
            sim.set_target("home", (rng.uniform(-1, 0), rng.uniform(-0.5, 0.5)))
            sim.set_target("away", (rng.uniform(0, 1), rng.uniform(-0.5, 0.5)))
        r = sim.step()
        st = sim.state()
        assert np.all(np.isfinite(sim.data.qpos)) and np.all(np.isfinite(sim.data.qvel))
        assert not r.out_of_bounds
        assert np.hypot(*st.puck_vel) < 12.0
        if r.goal:
            sim.reset(puck_pos=(0.0, 0.0), puck_vel=(rng.uniform(-3, 3), rng.uniform(-2, 2)))


@pytest.mark.parametrize(
    ("vx", "y", "expected"), [(4.0, 0.0, "home"), (-4.0, 0.05, "away"), (4.0, 0.35, None)]
)
def test_goal_detection(sim: HockeySim, vx: float, y: float, expected: str | None) -> None:
    sim.reset(puck_pos=(0.0, y), puck_vel=(vx, 0.0), home_pos=(-0.9, 0.4), away_pos=(0.9, -0.4))
    goal = None
    for _ in range(80):
        r = sim.step()
        if r.goal:
            goal = r.goal
            break
    assert goal == expected


def test_physics_controller_strikes_stationary_puck(sim: HockeySim, geom) -> None:
    from hockey.controllers.base import apply_command
    from hockey.observation import player_view

    ctrl = PhysicsController(geom)
    ctrl.reset(np.random.default_rng(0))
    sim.reset(puck_pos=(-0.45, 0.1), puck_vel=(0.0, 0.0))
    hit = False
    for _ in range(100):
        cmd = ctrl.act(player_view(sim.state(), "home"))
        apply_command(sim, "home", cmd)
        r = sim.step()
        hit |= any(e.kind == "home" for e in r.events)
        if hit and not sim._in_contact["home"]:
            break
    st = sim.state()
    assert hit
    assert st.puck_vel[0] > 1.5  # sent towards the opponent at speed


def test_fixed_seed_reproducibility() -> None:
    def run() -> tuple[np.ndarray, dict]:
        sim = HockeySim(SimConfig())
        from hockey.observation import Geometry

        g = Geometry.from_sim_config(sim.cfg)
        res = play_match(
            sim,
            PhysicsController(g),
            make_opponent("aggressive", g),
            seed=7,
            game=GameConfig(match_duration=8),
        )
        return sim.data.qpos.copy(), res.summary()

    q1, s1 = run()
    q2, s2 = run()
    assert np.array_equal(q1, q2)
    for k in ("goals_for", "goals_against", "hits_home", "hits_away"):
        assert s1[k] == s2[k]
