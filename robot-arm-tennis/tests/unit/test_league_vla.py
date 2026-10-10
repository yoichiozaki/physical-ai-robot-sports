"""Self-play league bookkeeping and the VLA task / planet-world machinery."""

import math

import numpy as np
import pytest

from tennis.control.ballistics import BallModel, BallPredictor
from tennis.learning.league import init_league, pfsp_weights, read_league, win_rate, write_league
from tennis.vla.tasks import COURT_X, in_court, in_zone, sample_task
from tennis.vla.worlds import WORLDS, sim_config


def test_pfsp_prefers_opponents_the_learner_loses_to(tmp_path) -> None:
    snaps = [
        {"name": "a", "points_won": 90, "points": 100},  # easy opponent
        {"name": "b", "points_won": 50, "points": 100},
        {"name": "c", "points_won": 10, "points": 100},  # hard opponent
        {"name": "d", "points_won": 0, "points": 0},  # unseen -> prior 0.5
    ]
    w = pfsp_weights(snaps, power=2.0)
    assert w.sum() == pytest.approx(1.0)
    assert w[2] > w[1] > w[0] > 0.0
    assert win_rate(snaps[3]) == pytest.approx(0.5)
    zip_path = tmp_path / "policy.zip"
    zip_path.write_bytes(b"dummy")
    league = init_league(tmp_path / "league", zip_path)
    assert (tmp_path / "league" / "gen_000.zip").exists()
    league["snapshots"][0]["points"] = 7
    write_league(tmp_path / "league", league)
    assert read_league(tmp_path / "league")["snapshots"][0]["points"] == 7


def test_planet_gravity_only_rescales_time() -> None:
    """Same launch geometry with speeds x s on a world with g s^2 -> same path, time / s."""
    p0 = np.array([-2.5, 0.2, 0.8])
    v0 = np.array([6.0, -0.4, 2.0])
    land = {}
    for name in ("earth", "moon"):
        w = WORLDS[name]
        bp = BallPredictor(BallModel(gravity=w.gravity), dt=0.002)
        pred = bp.predict(p0, v0 * w.time_scale, horizon=6.0, max_bounces=1)
        b = pred.first("bounce")
        land[name] = (np.array(b.pos), b.t * w.time_scale)
    assert np.linalg.norm(land["earth"][0] - land["moon"][0]) < 0.01
    assert land["earth"][1] == pytest.approx(land["moon"][1], rel=0.01)
    assert WORLDS["moon"].time_scale == pytest.approx(math.sqrt(1.62 / 9.81))


def test_world_gravity_reaches_mujoco() -> None:
    from tennis.sim.physics import TennisSim

    sim = TennisSim(sim_config(WORLDS["mars"]))
    assert sim.model.opt.gravity[2] == pytest.approx(-3.71)


def test_tasks_are_deterministic_and_consistent() -> None:
    a = sample_task("earth", 9.81, 123)
    b = sample_task("earth", 9.81, 123)
    assert a == b
    kinds = {sample_task("moon", 1.62, s).kind for s in range(60)}
    assert kinds == {"explicit", "away", "at"}
    for s in range(60):
        t = sample_task("earth", 9.81, s)
        if t.kind == "away":  # aim at the side the opponent is not on
            assert np.sign(t.target[1]) == -np.sign(t.opp_y)
        assert in_zone(np.array([*t.target, 0.033]), t.zone)
    assert not in_court(np.array([COURT_X + 0.1, 0.0, 0.0]))
    assert not in_court(None)


def test_recorded_expert_actions_replay_losslessly() -> None:
    """The 50 Hz (q, qd) action format reproduces the expert's stroke open-loop."""
    from tennis.vla.episode import ExpertController, ReplayController, run_episode, world_sim

    sim = world_sim("earth")
    expert = ExpertController("earth", sim)
    task = sample_task("earth", 9.81, 1003)
    rec = run_episode(task, expert, record=True)
    assert rec.hit and rec.frames
    actions = np.stack([f["action"] for f in rec.frames])
    assert actions.shape[1] == 18
    rep = run_episode(task, ReplayController(actions, expert.player.ready_q))
    assert rep.hit == rec.hit
    assert rep.in_zone == rec.in_zone
    assert np.linalg.norm(rep.landing[:2] - rec.landing[:2]) < 0.15


@pytest.mark.parametrize("latency", [0.0, 0.1, 0.24])
def test_chunk_execution_is_time_aligned(latency) -> None:
    """An oracle returning the expert's true future actions must reproduce the expert at any
    simulated latency (chunks are indexed from their observation time, stale steps skipped)."""
    from tennis.vla.episode import ChunkPolicyController, ExpertController, run_episode, world_sim

    sim = world_sim("earth")
    expert = ExpertController("earth", sim)
    task = sample_task("earth", 9.81, 1003)
    rec = run_episode(task, expert, record=True)
    acts = np.stack([f["action"] for f in rec.frames])
    dt = sim.cfg.physics.control_dt

    def oracle(obs: dict) -> np.ndarray:
        k = round(obs["time"] / dt)
        idx = np.clip(np.arange(k, k + 50), 0, len(acts) - 1)
        return acts[idx]

    res = run_episode(task, ChunkPolicyController(oracle, expert.player.ready_q, latency),
                      render=lambda s: np.zeros((8, 8, 3), np.uint8))  # fmt: skip
    assert res.hit and res.in_zone == rec.in_zone
    assert np.linalg.norm(res.landing[:2] - rec.landing[:2]) < 0.15
