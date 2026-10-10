"""Environment, matches, evaluation runner and configuration loading."""

import itertools

import numpy as np
import pytest

from tennis.config import from_dict
from tennis.evaluation.experiments import EvalExperimentConfig, run_one
from tennis.learning.env import EnvConfig, TennisEnv, decode_action


@pytest.mark.parametrize("mode", ["hybrid", "direct"])
def test_env_passes_gymnasium_checker(mode) -> None:
    from gymnasium.utils.env_checker import check_env

    env = TennisEnv(EnvConfig(mode=mode, points_per_episode=1))
    check_env(env, skip_render_check=True)
    obs, info = env.reset(seed=3)
    assert obs.shape == env.observation_space.shape
    total, done = 0.0, False
    for _ in range(40):
        obs, r, term, trunc, info = env.step(env.action_space.sample())
        total += r
        if term or trunc:
            done = True
            break
    assert done and "is_success" in info
    assert -1.0 <= total <= 1.0 + 0.1 * 40


def test_decode_action_ranges() -> None:
    shot = decode_action("hybrid", np.array([2, 2, 2]))
    assert shot.target == (2.6, 0.8) and shot.speed == 7.0
    cmd = decode_action("direct", np.array([-1.0, 1.0, 0.0]))
    assert cmd.pitch == pytest.approx(-0.1) and cmd.yaw == pytest.approx(0.45)
    assert cmd.speed == pytest.approx(5.0)


def test_match_is_deterministic_and_reports_metrics() -> None:
    task = {"controller": "B", "spec": "physics", "opponent": "steady", "condition": "nominal",
            "seed": 7, "points": 2}  # fmt: skip
    a, b = run_one(task), run_one(task)
    for k in ("points_won", "points_lost", "shots", "avg_rally_hits"):
        assert a[k] == b[k] or (np.isnan(a[k]) and np.isnan(b[k]))
    assert a["points_won"] + a["points_lost"] == 2
    assert 0.0 <= a["point_win_rate"] <= 1.0


def test_adaptive_controller_reports_identification() -> None:
    task = {"controller": "B*", "spec": "physics_adaptive", "opponent": "steady",
            "condition": "unknown_court", "seed": 3, "points": 2}  # fmt: skip
    row = run_one(task)
    assert "id_e" in row and row["id_bounces"] >= 1


def test_points_are_separated_and_every_serve_is_struck() -> None:
    """Regression: without a pause a robot pulled wide could not reach its own toss."""
    from tennis.control.factory import make_controller
    from tennis.control.players import opponent_factory
    from tennis.evaluation.match import Match
    from tennis.sim.game import GameConfig
    from tennis.sim.physics import TennisSim

    sim = TennisSim()
    home = make_controller("physics_rule", sim.cfg)
    away = opponent_factory(sim.cfg)["steady"]()
    game = GameConfig(points=6, max_point_time=8.0)
    m = Match(sim, home, away, seed=11, game=game)
    m.run()
    pts = m.ref.points
    for prev, nxt in itertools.pairwise(pts):
        assert nxt.start - prev.end >= game.point_gap - 0.03
    assert all(p.reason != "serve not hit" for p in pts)
    serves = [s for s in m.ref.shot_log if s.get("serve")]
    assert sum(s["landing"] in ("in", "volleyed") for s in serves) >= len(serves) - 1


def test_config_rejects_unknown_keys() -> None:
    with pytest.raises(ValueError):
        from_dict(EvalExperimentConfig, {"output": "x", "typo_key": 1})
    cfg = from_dict(EvalExperimentConfig, {"controllers": [{"name": "B", "spec": "physics"}]})
    assert cfg.controllers[0].spec == "physics"


def test_service_box_rule_keeps_serves_short_and_creates_rallies() -> None:
    from tennis.control.factory import make_controller
    from tennis.evaluation.match import Match
    from tennis.sim.game import GameConfig
    from tennis.sim.physics import TennisSim

    sim = TennisSim()
    a = make_controller("physics_rule", sim.cfg, "a")
    b = make_controller("physics_rule", sim.cfg, "b")
    game = GameConfig(points=4, max_point_time=8.0, service_line=1.62)
    m = Match(sim, a, b, seed=21, game=game)
    m.run()
    serves = [s for s in m.ref.shot_log if s.get("serve") and s["pos"] is not None]
    assert serves
    for s in serves:  # world frame: the receiver's half is the other sign of x
        assert abs(s["pos"][0]) <= 1.62 + sim.cfg.court.line_width / 2
    assert all(p.reason != "serve out of the service box" for p in m.ref.points)
    assert a.service_line == b.service_line == 1.62
