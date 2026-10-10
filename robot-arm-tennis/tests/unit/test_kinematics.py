"""Arm kinematics: closed-form IK, FK and agreement with the MuJoCo model."""

import mujoco
import numpy as np

from tennis.control.planner import SwingPlanner
from tennis.sim.kinematics import player_to_world, world_to_player


def _random_reachable(kin, rng, n=40):
    out = []
    while len(out) < n:
        gx, gy = rng.uniform(0.1, 1.3), rng.uniform(-0.7, 1.6)
        p = kin.shoulder(gx, gy) + np.array([rng.uniform(0.3, 0.7), rng.uniform(-0.6, 0.2),
                                             rng.uniform(-0.45, 0.1)])  # fmt: skip
        nrm = np.array([1.0, rng.uniform(-0.4, 0.4), rng.uniform(-0.3, 0.5)])
        nrm /= np.linalg.norm(nrm)
        q = kin.ik(p, nrm, (gx, gy))
        if q is not None:
            out.append((p, nrm, q))
    return out


def test_ik_fk_roundtrip(cfg, rng) -> None:
    kin = SwingPlanner(cfg).kin
    for p, nrm, q in _random_reachable(kin, rng):
        pos, rot, _ = kin.fk(q)
        assert np.allclose(pos, p, atol=1e-9)
        assert np.allclose(rot[:, 2], nrm, atol=1e-9)
        assert np.all(q[2:] >= kin.lo) and np.all(q[2:] <= kin.hi)


def test_fk_matches_mujoco_for_both_robots(sim, rng) -> None:
    kin = SwingPlanner(sim.cfg).kin
    for player in ("home", "away"):
        for _, _, q in _random_reachable(kin, rng, n=10):
            sim.data.qpos[sim._qadr[player]] = q
            mujoco.mj_forward(sim.model, sim.data)
            pos_root, rot_root, _ = kin.fk(q)
            pos_world = player_to_world(player, kin.root_to_player(pos_root))
            site = sim._site[player]
            assert np.allclose(sim.data.site_xpos[site], pos_world, atol=1e-9)
            n_world = player_to_world(player, rot_root[:, 2])
            assert np.allclose(sim.data.site_xmat[site].reshape(3, 3)[:, 2], n_world, atol=1e-9)


def test_unreachable_targets_are_rejected(cfg) -> None:
    kin = SwingPlanner(cfg).kin
    far = kin.shoulder(0.5, 0.0) + np.array([3.0, 0.0, 0.0])
    assert kin.ik(far, np.array([1.0, 0.0, 0.0]), (0.5, 0.0)) is None


def test_player_frames_are_involutions() -> None:
    v = np.array([1.0, -2.0, 0.5])
    assert np.allclose(world_to_player("home", v), v)
    assert np.allclose(world_to_player("away", world_to_player("away", v)), v)
    assert np.allclose(world_to_player("away", v), [-1.0, 2.0, 0.5])
