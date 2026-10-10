"""Unit tests: measurement filter, online adaptation, event-driven decisions, Colab job gating."""

from __future__ import annotations

import subprocess
import sys
import tarfile

import numpy as np
import pytest

from hockey.controllers.adaptation import OnlinePuckEstimator, PuckFilter
from hockey.controllers.prediction import PuckModel
from hockey.learning.env import AirHockeyEnv, EnvConfig
from hockey.observation import player_view
from hockey.simulation import HockeySim, PhysicsParams


def test_puck_filter_smooths_noise_and_resets_on_impact() -> None:
    f = PuckFilter(PuckModel(decay=0.0))
    rng = np.random.default_rng(0)
    p, v = np.array([0.0, 0.0]), np.array([1.0, 0.5])
    errs_raw, errs_f = [], []
    for k in range(50):
        t = k * 0.02
        meas_v = v + rng.normal(0, 0.05, 2)
        _, vf = f.update(t, p + v * t + rng.normal(0, 0.003, 2), meas_v)
        if k > 5:
            errs_raw.append(np.linalg.norm(meas_v - v))
            errs_f.append(np.linalg.norm(vf - v))
    assert np.mean(errs_f) < 0.7 * np.mean(errs_raw)
    # an impact (velocity reversal) is followed immediately, not smoothed away
    _, vf = f.update(1.0, p + v * 1.0, np.array([-1.0, 0.5]))
    assert vf[0] == pytest.approx(-1.0)


def test_online_estimator_identifies_table_parameters(geom) -> None:
    sim = HockeySim()
    sim.set_params(PhysicsParams(puck_decay=0.45, wall_restitution=0.70, mallet_restitution=0.7))
    est = OnlinePuckEstimator(geom, PuckModel(decay=0.15, wall_restitution=0.85))
    rng = np.random.default_rng(1)
    for _ in range(12):
        ang = rng.uniform(0.3, 1.2) * rng.choice([-1, 1])
        sim.reset(
            puck_pos=(rng.uniform(-0.3, 0.3), 0.0),
            puck_vel=(2.5 * np.cos(ang), 2.5 * np.sin(ang)),
            home_pos=(-0.95, 0.45),
            away_pos=(0.95, -0.45),
        )
        for _ in range(60):
            r = sim.step()
            est.update(player_view(sim.state(), "home"))
            if r.goal or r.mallet_contact:
                break
    m = est.model
    assert est.n_segments >= 5 and est.n_bounces >= 3
    assert abs(m.decay - 0.45) < abs(0.15 - 0.45) * 0.5  # moved most of the way to the truth
    assert abs(m.wall_restitution - 0.70) < 0.06


def test_hybrid_env_decisions_are_event_driven() -> None:
    env = AirHockeyEnv(
        EnvConfig(action_mode="hybrid", stage=4, decision_min_steps=5, decision_max_steps=100)
    )
    env.reset(seed=3)
    steps = []
    for _ in range(200):
        _, _, term, trunc, info = env.step(np.array([1, 2, 1]))
        steps.append(info["control_steps"])
        if term or trunc:
            env.reset()
    assert min(steps) >= 1 and max(steps) <= 100
    assert np.mean(steps) > 5  # a decision usually spans a whole shot


def test_colab_job_pack_plan_and_approval_gate(tmp_path) -> None:
    from hockey.infrastructure.colab import ColabJob, pack, plan

    bundle = pack(tmp_path / "bundle.tar.gz")
    with tarfile.open(bundle) as tar:
        names = tar.getnames()
    assert "physical-ai-hockey/src/hockey/__init__.py" in names
    assert not any("checkpoints" in n or ".env" in n for n in names)
    cmds = "\n".join(plan(ColabJob(hours=0.5)))
    for expected in (
        "colab --auth oauth2 new -s hockey --gpu T4",
        "colab drivemount -s hockey",
        "colab stop -s hockey",
    ):
        assert expected in cmds
    r = subprocess.run(
        [sys.executable, "scripts/colab_job.py", "run", "--hours", "0.5"],
        capture_output=True,
        text=True,
    )
    assert r.returncode == 2 and "approval" in r.stderr
    cmd = [sys.executable, "scripts/colab_job.py", "run", "--hours", "4", "--i-approve-auth"]
    r = subprocess.run([*cmd, "--i-approve-cu", "1"], capture_output=True, text=True)
    assert r.returncode == 2 and "exceeds the approved budget" in r.stderr
