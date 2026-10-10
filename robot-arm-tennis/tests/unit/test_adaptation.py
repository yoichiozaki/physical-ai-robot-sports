"""Online identification (ModelAdapter / ImpactAdapter) on synthetic observations."""

from dataclasses import replace
from types import SimpleNamespace

import numpy as np

from tennis.control.ballistics import BallModel, BallPredictor
from tennis.learning.adaptation import ImpactAdapter, ModelAdapter


class _Stub:
    def __init__(self) -> None:
        self.predictor = BallPredictor(BallModel(), dt=0.005)

    def set_model(self, model: BallModel) -> None:
        self.predictor.set_model(model)


def _feed(adapter: ModelAdapter, player: _Stub, true: BallModel, rng, shots: int = 8,
          noise: tuple[float, float] = (0.004, 0.05)) -> None:  # fmt: skip
    bp = BallPredictor(true, dt=0.005)
    t = 0.0
    for k in range(shots):
        p0 = np.array([-2.6, rng.uniform(-0.6, 0.6), rng.uniform(0.6, 1.0)])
        v0 = np.array([rng.uniform(5.0, 7.0), rng.uniform(-0.8, 0.8), rng.uniform(0.5, 2.0)])
        pred = bp.predict(p0, v0, horizon=1.6, max_bounces=2)
        for i in range(0, len(pred.t), 4):  # 50 Hz
            view = SimpleNamespace(
                time=t + pred.t[i], ball_pos=pred.pos[i] + rng.normal(0, noise[0], 3),
                ball_vel=pred.vel[i] + rng.normal(0, noise[1], 3), ball_id=k, point_live=True,
            )  # fmt: skip
            adapter.observe(view, player)
        t += 3.0


def test_identifies_a_lively_slippery_court(rng) -> None:
    player, ad = _Stub(), ModelAdapter()
    ad.reset(player)
    true = replace(BallModel(), bounce_e=0.86, bounce_mu=0.10)
    _feed(ad, player, true, rng)
    assert ad.n_bounces >= 5
    assert abs(ad.e - 0.86) < 0.04
    assert ad.court_shifted
    assert abs(player.predictor.model.bounce_e - ad.e) < 1e-12  # identified model in use


def test_identifies_drag_and_cross_wind(rng) -> None:
    player, ad = _Stub(), ModelAdapter()
    ad.reset(player)
    true = replace(BallModel(), k_drag=0.032, wind=(0.0, 2.2, 0.0))
    _feed(ad, player, true, rng, shots=14)
    assert ad.air_shifted
    assert abs(ad.k_drag - 0.032) / 0.032 < 0.3
    assert abs(ad.wind[1] - 2.2) < 0.7


def test_nominal_conditions_keep_the_nominal_model(rng) -> None:
    player, ad = _Stub(), ModelAdapter()
    ad.reset(player)
    _feed(ad, player, BallModel(), rng)
    assert not ad.shifted
    assert player.predictor.model == BallModel()


def test_impact_gain_shrinks_towards_observations() -> None:
    from tennis.control.planner import SwingPlanner
    from tennis.sim.config import TennisConfig

    planner = SwingPlanner(TennisConfig())
    ia = ImpactAdapter()
    ia.reset(planner)
    a0 = planner.pc.impact_a
    n = np.array([1.0, 0.0, 0.0])
    for _ in range(8):  # the ball always leaves 20 % faster than planned
        plan = SimpleNamespace(v_out=np.array([5.0, 0.0, 1.0]), normal=n)
        ia.observe(plan, np.array([6.0, 0.0, 1.0]), 0.0, 0.0, planner)
    assert ia.shifted and 1.1 < ia.k < 1.2
    assert abs(planner.pc.impact_a - a0 * ia.k) < 1e-12
