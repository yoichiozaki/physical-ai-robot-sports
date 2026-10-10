"""Integration tests: Gymnasium environment, matches, training/reload, rendering."""

from __future__ import annotations

import numpy as np
import pytest
from gymnasium.utils.env_checker import check_env

from hockey.controllers.factory import make_controller
from hockey.controllers.opponents import OPPONENT_NAMES, make_opponent
from hockey.evaluation.match import play_match
from hockey.learning.env import AirHockeyEnv, EnvConfig
from hockey.simulation import HockeySim
from hockey.simulation.game import GameConfig


@pytest.mark.parametrize("mode", ["direct", "hybrid"])
def test_env_passes_gymnasium_checker(mode: str) -> None:
    env = AirHockeyEnv(EnvConfig(action_mode=mode, stage=4))
    check_env(env, skip_render_check=True)
    env.close()


@pytest.mark.parametrize("mode", ["direct", "hybrid"])
def test_env_shapes_bounds_and_termination(mode: str) -> None:
    env = AirHockeyEnv(EnvConfig(action_mode=mode, stage=4))
    obs, _ = env.reset(seed=0)
    assert obs.shape == env.observation_space.shape
    episodes = 0
    for _ in range(1500):
        a = env.action_space.sample()
        assert env.action_space.contains(a)
        obs, r, term, trunc, info = env.step(a)
        assert env.observation_space.contains(obs)
        assert np.isfinite(r) and np.all(np.isfinite(obs))
        if term or trunc:
            episodes += 1
            assert "is_success" in info
            env.reset()
    assert episodes >= 5
    env.close()


def test_env_seed_reproducibility() -> None:
    def rollout(seed: int) -> list[float]:
        env = AirHockeyEnv(EnvConfig(stage=4))
        env.reset(seed=seed)
        env.action_space.seed(seed)
        out = []
        for _ in range(200):
            obs, r, term, trunc, _ = env.step(env.action_space.sample())
            out.append(float(r) + float(obs.sum()))
            if term or trunc:
                env.reset()
        return out

    assert rollout(5) == rollout(5)
    assert rollout(5) != rollout(6)


def test_rule_based_policies_run_in_env() -> None:
    """Rule-based policies through both action interfaces run many episodes.

    * hybrid interface: a constant "direct shot at the goal centre" action is exactly the
      physics-based baseline and must score in a fair share of episodes;
    * direct interface: the reactive baseline's target mapped to the Box action.
    """
    from hockey.controllers.reactive import ReactiveController

    env = AirHockeyEnv(EnvConfig(action_mode="hybrid", action_repeat=5, stage=4))
    env.reset(seed=1)
    action = np.array([1, 2, 1])  # DIRECT, aim 0.0, 3.5 m/s
    successes = episodes = 0
    while episodes < 25:
        _, _, term, trunc, info = env.step(action)
        if term or trunc:
            episodes += 1
            successes += int(info["is_success"])
            env.reset()
    assert successes >= 3

    env = AirHockeyEnv(EnvConfig(stage=4))
    ctrl = ReactiveController(env.geom)
    env.reset(seed=1)
    (x0, x1), (y0, y1) = env.geom.x_range, env.geom.y_range
    episodes = 0
    for _ in range(2000):
        tgt = ctrl.act(env._last_view).target
        a = np.array([(tgt[0] - x0) / (x1 - x0) * 2 - 1, (tgt[1] - y0) / (y1 - y0) * 2 - 1])
        _, _, term, trunc, info = env.step(np.clip(a, -1, 1).astype(np.float32))
        if term or trunc:
            episodes += 1
            env.reset()
    assert episodes >= 5


def test_match_runs_against_all_opponents(geom) -> None:
    sim = HockeySim()
    for opp in OPPONENT_NAMES:
        res = play_match(
            sim,
            make_controller("physics", geom),
            make_opponent(opp, geom),
            0,
            game=GameConfig(match_duration=10),
        )
        s = res.summary()
        assert s["anomalies"] == 0
        assert s["duration_s"] >= 10.0 - 1e-6
        assert s["latency_ms_mean"] < 20.0


@pytest.mark.slow
def test_training_checkpoint_reload_and_evaluate(tmp_path, geom) -> None:
    from hockey.learning.train import load_train_config, train

    cfg = load_train_config("configs/training/smoke.yaml", {"total_timesteps": 2048, "n_envs": 1})
    run_dir = train(cfg, tmp_path)
    model_path = run_dir / "final_model.zip"
    assert model_path.exists() and (run_dir / "run_info.json").exists()
    ctrl = make_controller(f"ppo:{model_path}", geom)
    res = play_match(
        HockeySim(), ctrl, make_opponent("random", geom), 0, game=GameConfig(match_duration=5)
    )
    assert res.summary()["anomalies"] == 0


@pytest.mark.render
def test_render_frame_with_overlays_and_video(tmp_path, geom) -> None:
    from hockey.controllers.physics_based import PhysicsController
    from hockey.observation import player_view
    from hockey.visualization.hud import HudState, draw_hud
    from hockey.visualization.renderer import SceneRenderer, decision_overlays
    from hockey.visualization.video import VideoWriter, probe

    sim = HockeySim()
    sim.reset(puck_pos=(0.3, 0.2), puck_vel=(-2.5, -1.2))
    ctrl = PhysicsController(geom)
    cmd = ctrl.act(player_view(sim.state(), "home"))
    with SceneRenderer(sim, 320, 180, "top") as r:
        frame = r.render(
            decision_overlays(cmd.info, "home", cmd.target, geom.puck_radius, geom.mallet_radius)
        )
        assert frame.shape == (180, 320, 3) and frame.std() > 5
        frame = draw_hud(frame, HudState(strategy="防御", caption="テスト"))
        with VideoWriter(tmp_path / "t.mp4", 320, 180, fps=10) as vw:
            for _ in range(10):
                vw.write(frame)
    info = probe(tmp_path / "t.mp4")
    assert info["codec"] == "h264" and info["duration"] == pytest.approx(1.0, abs=0.15)
