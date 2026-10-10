"""Swing planning pieces: impact model, trajectories, candidates."""

import numpy as np

from tennis.control.planner import SwingPlanner
from tennis.sim.trajectory import JointTrajectory


def test_impact_model_inverse_is_exact(cfg, rng) -> None:
    pl = SwingPlanner(cfg)
    for _ in range(50):
        v_in = np.array([rng.uniform(-8, -3), rng.uniform(-1.5, 1.5), rng.uniform(-3, 2)])
        v_out = np.array([rng.uniform(3, 8), rng.uniform(-1.5, 1.5), rng.uniform(0.5, 3)])
        n, s = pl.impact_inverse(v_in, v_out)
        assert abs(np.linalg.norm(n) - 1) < 1e-9
        assert np.allclose(pl.impact_forward(v_in, n, s), v_out, atol=1e-9)


def test_appended_segments_keep_velocity_continuous() -> None:
    q0 = np.zeros(8)
    traj = JointTrajectory.hold(q0, 0.0)
    traj.append(np.full(8, 0.3), np.full(8, 1.2), 0.5)  # ends moving (swing through contact)
    traj.append(np.full(8, 0.5), np.zeros(8), 0.4)  # follow-through starts from that motion
    eps = 1e-6
    _, v_a, _ = traj.sample(0.5 - eps)
    _, v_b, _ = traj.sample(0.5 + eps)
    assert np.allclose(v_a, 1.2, atol=1e-4)
    assert np.allclose(v_a, v_b, atol=1e-4)  # regression: follow-through used to start at rest
    q_end, v_end, _ = traj.sample(10.0)
    assert np.allclose(q_end, 0.5) and np.allclose(v_end, 0.0)


def test_planner_produces_contact_on_time(cfg) -> None:
    from tennis.control.ballistics import BallModel, BallPredictor
    from tennis.control.planner import ShotSpec, contact_candidates
    from tennis.control.players import PhysicsPlayer

    player = PhysicsPlayer(cfg)
    pl = player.planner
    bp = BallPredictor(BallModel(), dt=0.005)
    pred = bp.predict(np.array([2.6, 0.0, 0.8]), np.array([-6.5, -0.4, 2.0]), 2.2, max_bounces=2)
    cands = contact_candidates(pred, pl.pc, pl.kin, pl.lo, pl.hi, False, False)
    assert cands, "an ordinary ball must have reachable contact candidates"
    start = player.traj.sample(0.0)
    plan = None
    for c in cands[: pl.pc.candidates]:
        plan = pl.plan(
            0.0, start, c.t, c.pos, c.vel, ShotSpec((2.0, 0.0), 5.5), player.shot_predictor
        )
        if plan is not None:
            break
    assert plan is not None
    q, _, _ = plan.trajectory.sample(plan.t_contact)
    pos, _, _ = pl.kin.fk(q)
    assert np.linalg.norm(pl.kin.root_to_player(pos) - plan.ball_pos) < 0.05
