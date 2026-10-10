"""Unit tests: puck prediction, planning, motion control, observation, config."""

from __future__ import annotations

import math

import numpy as np
import pytest

from hockey.config import from_dict, load_yaml
from hockey.controllers.planning import (
    PlannerConfig,
    ShotSpec,
    Strategy,
    StrikePlanner,
    path_clearance,
    shot_direction,
    shot_path,
)
from hockey.controllers.prediction import PuckModel, PuckPredictor
from hockey.learning.train import TrainConfig
from hockey.observation import OBS_DIM, encode_observation, player_view, to_world
from hockey.simulation import HockeySim, PhysicsParams
from hockey.simulation.robot import SOFT_MARGIN, AxisLimits, MotionController, min_time_1d


def test_prediction_straight_line_with_decay(geom) -> None:
    p = PuckPredictor(geom, PuckModel(decay=0.2, wall_restitution=0.85))
    traj = p.predict(np.array([0.0, 0.0]), np.array([1.0, 0.0]), horizon=0.5)
    x, y, vx, vy = traj.state_at(0.5)
    assert x == pytest.approx((1 - math.exp(-0.1)) / 0.2, rel=1e-9)
    assert vx == pytest.approx(math.exp(-0.1), rel=1e-9)
    assert y == 0.0 and vy == 0.0


def test_prediction_reflection_and_goal(geom) -> None:
    p = PuckPredictor(geom, PuckModel(decay=0.0, wall_restitution=0.8))
    traj = p.predict(np.array([0.0, 0.0]), np.array([0.0, 2.0]), horizon=1.0)
    assert traj.segments[0].end == "wall_side"
    ylim = geom.half_width - geom.puck_radius
    t_hit = ylim / 2.0
    _, y, _, vy = traj.state_at(t_hit + 0.1)
    assert vy == pytest.approx(-1.6)
    assert y == pytest.approx(ylim - 0.16)
    goal = p.predict(np.array([0.0, 0.0]), np.array([-3.0, 0.0]), horizon=2.0)
    assert goal.end_event == "goal_own"
    hit = goal.first_crossing_x(-0.8)
    assert hit is not None and hit[0] == pytest.approx(0.8 / 3.0)


def test_prediction_matches_mujoco_free_flight(sim: HockeySim, geom) -> None:
    """The analytic model reproduces simulated flights incl. a wall bounce within 2 cm."""
    sim.set_params(PhysicsParams(puck_decay=0.15, wall_restitution=0.85, mallet_restitution=0.7))
    pred = PuckPredictor(geom, PuckModel(decay=0.15, wall_restitution=0.85))
    rng = np.random.default_rng(3)
    for _ in range(5):
        p0 = np.array([rng.uniform(-0.3, 0.3), rng.uniform(-0.2, 0.2)])
        v0 = np.array([rng.uniform(-1.5, 1.5), rng.choice([-1, 1]) * rng.uniform(1.0, 2.5)])
        sim.reset(puck_pos=p0, puck_vel=v0, home_pos=(-0.95, 0.45), away_pos=(0.95, -0.45))
        traj = pred.predict(p0, v0, horizon=0.6)
        for _ in range(30):
            sim.step()
        x, y, _, _ = traj.state_at(0.6)
        assert math.dist((x, y), sim.state().puck_pos) < 0.02


def test_min_time_1d_properties() -> None:
    assert min_time_1d(0.0, 0.0, 3.0, 20.0) == 0.0
    t = min_time_1d(0.1, 0.0, 3.0, 20.0)
    assert t == pytest.approx(math.sqrt(2 * 0.1 / 20.0))
    assert min_time_1d(1.0, 0.0, 3.0, 20.0) == pytest.approx(3.0 / 20.0 + (1.0 - 9.0 / 40.0) / 3.0)
    assert (
        min_time_1d(0.5, -2.0, 3.0, 20.0)
        > min_time_1d(0.5, 0.0, 3.0, 20.0)
        > min_time_1d(0.5, 2.0, 3.0, 20.0)
    )


def test_motion_controller_limits() -> None:
    lim = (AxisLimits(-1.0, 0.0, 3.0, 15.0), AxisLimits(-0.5, 0.5, 3.0, 25.0))
    mc = MotionController(*lim)
    q = np.array([-0.5, 0.0])
    dt = 0.001
    u_prev = np.zeros(2)
    for _ in range(2000):
        u = np.array(mc.command(5.0, -5.0, q[0], q[1], dt))
        assert abs(u[0]) <= 3.0 + 1e-9 and abs(u[1]) <= 3.0 + 1e-9
        assert abs(u[0] - u_prev[0]) <= 15.0 * dt + 1e-9
        assert abs(u[1] - u_prev[1]) <= 25.0 * dt + 1e-9
        q += u * dt
        u_prev = u
    assert mc.clipped_last
    assert q[0] == pytest.approx(0.0 - SOFT_MARGIN, abs=2e-3) and q[1] == pytest.approx(
        -0.5 + SOFT_MARGIN, abs=2e-3
    )


def test_bank_shot_direction_reaches_goal(geom) -> None:
    pred = PuckPredictor(geom, PuckModel(decay=0.15, wall_restitution=0.85))
    for strat in (Strategy.BANK_LEFT, Strategy.BANK_RIGHT, Strategy.DIRECT):
        spec = ShotSpec(strat, 0.05, 3.5)
        d = shot_direction(-0.4, 0.1, spec, geom, 0.85)
        assert d is not None and d[0] > 0
        traj = pred.predict(np.array([-0.4, 0.1]), np.array(d) * 3.5, horizon=2.0)
        assert traj.end_event == "goal_opp"
        x, y, vx, vy = traj.state_at(traj.end_time)
        y_goal_line = y + (geom.half_length - x) * vy / vx  # aim point is defined on the goal line
        assert y_goal_line == pytest.approx(0.05, abs=0.005)
        if strat != Strategy.DIRECT:
            assert len(traj.bounce_points()) == 1


def test_path_clearance(geom) -> None:
    path = shot_path(-0.4, 0.0, ShotSpec(Strategy.DIRECT, 0.0), geom, 0.85)
    assert path_clearance(path, 0.8, 0.0) == pytest.approx(0.0)
    assert path_clearance(path, 0.8, 0.3) == pytest.approx(0.3)


def test_planner_rejects_unreachable_and_plans_reachable(geom) -> None:
    planner = StrikePlanner(geom, PlannerConfig())
    limits = geom.axis_limits()
    pred = PuckPredictor(geom, PuckModel())
    traj = pred.predict(np.array([-0.5, 0.0]), np.array([0.0, 0.0]), horizon=1.5)
    plan = planner.find_strike(traj, ShotSpec(), np.array([-0.84, 0.0]), np.zeros(2), limits)
    assert plan is not None and plan.slack >= 0
    nx, ny = plan.normal
    assert (
        math.dist(
            plan.mallet_contact,
            (
                -0.5 - (geom.puck_radius + geom.mallet_radius) * nx,
                -(geom.puck_radius + geom.mallet_radius) * ny,
            ),
        )
        < 1e-9
    )
    # a puck arriving almost immediately on the far side of the table cannot be struck in time
    plan2 = planner.plan_at(
        0.02, (-0.2, 0.4), (-3.0, 0.0), ShotSpec(), np.array([-0.84, -0.4]), np.zeros(2), limits
    )
    assert plan2 is None or plan2.slack < 0


def test_player_view_mirroring(sim: HockeySim, geom) -> None:
    sim.reset(puck_pos=(0.3, 0.2), puck_vel=(1.0, -0.5), home_pos=(-0.5, 0.1), away_pos=(0.6, -0.2))
    st = sim.state()
    away = player_view(st, "away")
    assert np.allclose(away.puck_pos, (-0.3, 0.2)) and np.allclose(away.puck_vel, (-1.0, -0.5))
    assert np.allclose(away.own_pos, (-0.6, -0.2)) and np.allclose(away.opp_pos, (0.5, 0.1))
    assert np.allclose(to_world("away", away.own_pos), st.away_pos)
    obs = encode_observation(player_view(st, "home"), geom)
    assert obs.shape == (OBS_DIM,) and obs.dtype == np.float32 and np.all(np.isfinite(obs))


def test_config_loading_rejects_unknown_keys() -> None:
    cfg = from_dict(TrainConfig, load_yaml("configs/training/ppo_hybrid.yaml"))
    assert cfg.env.action_mode == "hybrid" and cfg.env.reward.goal_scored == 10.0
    with pytest.raises(ValueError, match="Unknown config keys"):
        from_dict(TrainConfig, {"not_a_key": 1})
