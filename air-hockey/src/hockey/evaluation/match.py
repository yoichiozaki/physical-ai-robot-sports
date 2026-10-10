"""Run a full match between two controllers and collect metrics."""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass, replace

import numpy as np

from hockey.controllers.base import Command, Controller, apply_command
from hockey.controllers.prediction import PuckModel
from hockey.evaluation.conditions import NOMINAL, ConditionConfig
from hockey.evaluation.tracker import MatchResult, MatchTracker
from hockey.observation import Geometry, SensorNoise, apply_noise, player_view
from hockey.simulation.game import GameConfig, Referee
from hockey.simulation.physics import HockeySim, PhysicsParams, StepResult


@dataclass
class StepRecord:
    """Everything a renderer / logger needs about one control step."""

    sim: HockeySim
    home_cmd: Command
    away_cmd: Command
    result: StepResult
    event: str | None
    score: tuple[int, int]
    time_remaining: float


def true_model(params: PhysicsParams, identified_mallet_ratio: float = 0.68 / 0.70) -> PuckModel:
    """Ground-truth puck model for metrics (simulator parameters)."""
    return PuckModel(
        decay=params.puck_decay,
        wall_restitution=params.wall_restitution,
        mallet_restitution=params.mallet_restitution * identified_mallet_ratio,
    )


def play_match(
    sim: HockeySim,
    home: Controller,
    away: Controller,
    seed: int,
    condition: ConditionConfig = NOMINAL,
    game: GameConfig | None = None,
    on_step: Callable[[StepRecord], None] | None = None,
    first: str | None = None,
    setup: Callable[[HockeySim], None] | None = None,
    stop_on_goal: bool = False,
) -> MatchResult:
    """Play one match (fixed game time) and return the home player's metrics.

    ``setup`` may override the initial face-off (e.g. a scripted shot for a demo clip);
    ``stop_on_goal`` ends the match at the first goal.
    """
    rng = np.random.default_rng(seed)
    params, noise = condition.sample(rng)
    params = replace(
        params, away_speed_scale=params.away_speed_scale * getattr(away, "speed_factor", 1.0)
    )
    sim.set_params(params)
    game = game or GameConfig()
    ref = Referee(sim, game, np.random.default_rng(rng.integers(2**31)))
    home.reset(np.random.default_rng(rng.integers(2**31)))
    away.reset(np.random.default_rng(rng.integers(2**31)))
    if hasattr(away, "set_speed_scale"):
        away.set_speed_scale(params.away_speed_scale)
    noise_rng = np.random.default_rng(rng.integers(2**31))
    geom = Geometry.from_sim_config(sim.cfg)
    tracker = MatchTracker(geom, true_model(params), sim.control_dt)
    ref.start(first)
    if setup is not None:
        setup(sim)
    no_noise = SensorNoise()
    while not ref.done:
        before = sim.state()
        score = (ref.score[0], ref.score[1])
        hv = apply_noise(player_view(before, "home", score, ref.time_remaining), noise, noise_rng)
        av = apply_noise(
            player_view(before, "away", score, ref.time_remaining), no_noise, noise_rng
        )
        t0 = time.perf_counter()
        hc = home.act(hv)
        latency = time.perf_counter() - t0
        ac = away.act(av)
        apply_command(sim, "home", hc)
        apply_command(sim, "away", ac)
        result = sim.step()
        after = sim.state()
        event = ref.update(result)
        tracker.update(before, result, after, hc.info, latency, event)
        if on_step is not None:
            on_step(
                StepRecord(
                    sim, hc, ac, result, event, (ref.score[0], ref.score[1]), ref.time_remaining
                )
            )
        if stop_on_goal and event in ("goal_home", "goal_away"):
            break
    return tracker.finish(sim.time, ref.anomalies, ref.possession_resets)
