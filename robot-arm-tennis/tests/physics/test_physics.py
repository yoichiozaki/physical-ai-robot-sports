"""Physical plausibility of the MuJoCo world (ball, court, servos)."""

import numpy as np

from tennis.sim.physics import PhysicsParams
from tennis.sim.trajectory import JointTrajectory


def _apex_after_bounce(sim, h0: float) -> float:
    sim.reset(ball_pos=(-1.5, 0.3, h0), ball_vel=(0.0, 0.0, 0.0))
    bounced, apex = False, 0.0
    while sim.time < 2.0:
        res = sim.step()
        if any(e.kind == "floor" for e in res.events):
            if bounced:
                break
            bounced = True
        z = sim.ball()[0][2]
        if bounced:
            apex = max(apex, z)
    return apex


def test_court_bounce_restitution(sim) -> None:
    sim.set_params(PhysicsParams(court_restitution=0.75))
    r = sim.cfg.ball.radius
    h0 = 1.0
    apex = _apex_after_bounce(sim, h0)
    e = np.sqrt((apex - r) / (h0 - r))  # drag makes this a slight underestimate
    assert 0.66 < e < 0.80


def test_livelier_court_bounces_higher(sim) -> None:
    sim.set_params(PhysicsParams(court_restitution=0.70))
    low = _apex_after_bounce(sim, 1.0)
    sim.set_params(PhysicsParams(court_restitution=0.86))
    high = _apex_after_bounce(sim, 1.0)
    assert high > low + 0.1


def test_ball_at_rest_stays_at_rest(sim) -> None:
    sim.set_params(PhysicsParams())
    r = sim.cfg.ball.radius
    sim.reset(ball_pos=(-1.0, 0.5, r), ball_vel=(0.0, 0.0, 0.0))
    for _ in range(50):
        sim.step()
    pos, vel = sim.ball()
    assert np.linalg.norm(pos - np.array([-1.0, 0.5, r])) < 2e-3
    assert np.linalg.norm(vel) < 1e-2


def test_servos_track_a_joint_trajectory(sim) -> None:
    from tennis.control.players import PhysicsPlayer

    sim.set_params(PhysicsParams())
    ready = PhysicsPlayer(sim.cfg).ready_q
    sim.reset(ball_pos=(0.0, 3.0, 0.033), robot_q={"home": ready, "away": ready})
    goal = ready + np.array([0.4, 0.3, 0.3, -0.2, 0.25, 0.3, -0.2, 0.4])
    traj = JointTrajectory.hold(ready, 0.0)
    traj.append(goal, np.zeros(8), 0.6)
    sim.set_trajectory("home", traj)
    worst = 0.0
    while sim.time < 0.9:
        sim.step()
        q_ref, _, _ = traj.sample(sim.time)
        worst = max(worst, float(np.abs(sim.state().q["home"] - q_ref).max()))
    assert worst < 0.05  # rad / m along the whole motion
    assert np.abs(sim.state().q["home"] - goal).max() < 0.01
