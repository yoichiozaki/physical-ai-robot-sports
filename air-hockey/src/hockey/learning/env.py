"""Gymnasium environment: the learning agent controls the *home* gantry robot."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any, ClassVar

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from hockey import ENV_VERSION
from hockey.controllers.base import Controller, DecisionInfo, apply_command
from hockey.controllers.opponents import StationaryOpponent, make_opponent
from hockey.controllers.prediction import PuckModel, PuckPredictor
from hockey.evaluation.conditions import CONDITIONS
from hockey.evaluation.match import true_model
from hockey.learning.actions import (
    DEFAULT_AIMS,
    DEFAULT_SPEEDS,
    DirectTargetAction,
    HybridStrategyAction,
)
from hockey.learning.curriculum import DEFAULT_STAGES, StageConfig, reset_scenario, sample_scenario
from hockey.learning.rewards import RewardConfig, RewardFunction
from hockey.observation import (
    OBS_DIM,
    Geometry,
    SideClock,
    apply_noise,
    encode_observation,
    player_view,
)
from hockey.simulation.config import SimConfig
from hockey.simulation.physics import HockeySim


@dataclass(frozen=True)
class EnvConfig:
    sim: SimConfig = field(default_factory=SimConfig)
    reward: RewardConfig = field(default_factory=RewardConfig)
    action_mode: str = "direct"  # "direct" | "hybrid"
    action_repeat: int = 2  # direct mode: control steps (20 ms each) per agent decision
    decision_min_steps: int = 5  # hybrid mode: event-driven decisions (see AirHockeyEnv.step)
    decision_max_steps: int = 100
    stages: tuple[StageConfig, ...] = DEFAULT_STAGES
    stage: int = 0  # index into ``stages``
    hybrid_aims: tuple[float, ...] = DEFAULT_AIMS
    hybrid_speeds: tuple[float, ...] = DEFAULT_SPEEDS
    defense_offset: float = 0.13  # x of the defence line (from own goal) used by rewards


class AirHockeyEnv(gym.Env):
    metadata: ClassVar[dict[str, Any]] = {"render_modes": ["rgb_array"], "render_fps": 50}

    def __init__(self, cfg: EnvConfig | None = None, render_mode: str | None = None) -> None:
        super().__init__()
        self.cfg = cfg or EnvConfig()
        self.render_mode = render_mode
        self.sim = HockeySim(self.cfg.sim)
        self.geom = Geometry.from_sim_config(self.cfg.sim)
        self.nominal_model = PuckModel()
        if self.cfg.action_mode == "direct":
            self.adapter: DirectTargetAction | HybridStrategyAction = DirectTargetAction(self.geom)
        elif self.cfg.action_mode == "hybrid":
            self.adapter = HybridStrategyAction(
                self.geom, self.nominal_model, self.cfg.hybrid_aims, self.cfg.hybrid_speeds
            )
        else:
            raise ValueError(f"Unknown action_mode {self.cfg.action_mode}")
        self.action_space = self.adapter.space
        dim = OBS_DIM + self.adapter.feature_dim
        self.observation_space = spaces.Box(-5.0, 5.0, shape=(dim,), dtype=np.float32)
        self.stage_index = self.cfg.stage
        self._opponents: dict[str, Controller] = {}
        self._drill_opponent = StationaryOpponent(self.geom)
        self.opponent: Controller = self._drill_opponent
        self.scenario = "strike"
        self.max_time = 5.0
        self.noise_rng = np.random.default_rng(0)
        self.noise = CONDITIONS["nominal"].sample(np.random.default_rng(0))[1]
        self.reward_fn: RewardFunction | None = None
        self.last_info = DecisionInfo()
        self._renderer = None
        self.episode_terms: dict[str, float] = {}
        self.clock = SideClock()

    # ------------------------------------------------------------------ curriculum
    @property
    def stage(self) -> StageConfig:
        return self.cfg.stages[min(self.stage_index, len(self.cfg.stages) - 1)]

    def set_stage(self, index: int) -> None:
        self.stage_index = int(index)

    def _opponent(self, name: str) -> Controller:
        if name not in self._opponents:
            self._opponents[name] = make_opponent(name, self.geom, self.nominal_model)
        return self._opponents[name]

    # ------------------------------------------------------------------ gym API
    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        super().reset(seed=seed)
        rng = self.np_random
        options = options or {}
        stage = self.stage
        cond = CONDITIONS[options.get("condition", stage.condition)]
        params, self.noise = cond.sample(rng)
        self.scenario = options.get("scenario") or sample_scenario(stage, rng)
        if self.scenario == "match":
            name = (
                options.get("opponent") or stage.opponents[int(rng.integers(len(stage.opponents)))]
            )
            self.opponent = self._opponent(name)
            self.max_time = stage.match_time
        else:
            self.opponent = self._drill_opponent
            self.max_time = stage.drill_time
        factor = getattr(self.opponent, "speed_factor", 1.0)
        params = replace(params, away_speed_scale=params.away_speed_scale * factor)
        self.sim.set_params(params)
        self.opponent.reset(np.random.default_rng(int(rng.integers(2**31))))
        if hasattr(self.opponent, "set_speed_scale"):
            self.opponent.set_speed_scale(params.away_speed_scale)
        self.adapter.reset(np.random.default_rng(int(rng.integers(2**31))))
        self.noise_rng = np.random.default_rng(int(rng.integers(2**31)))
        reset_scenario(self.sim, self.scenario, stage, rng)
        truth = PuckPredictor(self.geom, true_model(params))
        self.reward_fn = RewardFunction(
            self.cfg.reward, truth, -self.geom.half_length + self.cfg.defense_offset, self.geom.vmax
        )
        state = self.sim.state()
        self.reward_fn.reset(state)
        self.episode_terms = {}
        self.clock.reset()
        view = self._home_view()
        self.clock.update(view)
        self._last_view = view
        obs = self._observation(view)
        return obs, {
            "scenario": self.scenario,
            "opponent": self.opponent.name,
            "env_version": ENV_VERSION,
        }

    def _home_view(self):
        state = self.sim.state()
        view = player_view(state, "home")
        return apply_noise(view, self.noise, self.noise_rng)

    def _observation(self, view) -> np.ndarray:
        base = encode_observation(view, self.geom, self.clock.value)
        if self.adapter.feature_dim:
            return np.concatenate([base, self.adapter.features(view)]).astype(np.float32)
        return base

    def step(self, action):
        """Advance until the next decision point.

        * direct mode: a fixed ``action_repeat`` control steps (fixed-rate decisions);
        * hybrid mode: event-driven — at least ``decision_min_steps`` control steps, then
          until the executor is idle again (the chosen shot was played) or
          ``decision_max_steps`` elapsed. One decision therefore corresponds to one shot.
        """
        assert self.reward_fn is not None, "call reset() first"
        self.adapter.set_action(action)
        total = 0.0
        terminated = truncated = False
        goal = None
        hybrid = self.cfg.action_mode == "hybrid"
        min_steps = self.cfg.decision_min_steps if hybrid else self.cfg.action_repeat
        max_steps = self.cfg.decision_max_steps if hybrid else self.cfg.action_repeat
        view = self._last_view
        steps = 0
        while True:
            before = self.sim.state()
            home_cmd = self.adapter.command(view)
            self.last_info = home_cmd.info
            opp_cmd = self.opponent.act(player_view(before, "away"))
            apply_command(self.sim, "home", home_cmd)
            apply_command(self.sim, "away", opp_cmd)
            result = self.sim.step()
            after = self.sim.state()
            u = self.sim.motion["home"].u
            r, terms = self.reward_fn(
                before, result, after, (u[0], u[1]), result.target_clipped["home"]
            )
            for k, v in terms.items():
                self.episode_terms[k] = self.episode_terms.get(k, 0.0) + v
            total += r
            steps += 1
            view = apply_noise(player_view(after, "home"), self.noise, self.noise_rng)
            self.clock.update(view)
            if result.goal is not None or result.out_of_bounds:
                terminated = True
                goal = result.goal
                break
            if after.time >= self.max_time - 1e-9:
                truncated = True
                break
            if steps >= max_steps or (steps >= min_steps and self.adapter.needs_decision(view)):
                break
        self._last_view = view
        obs = self._observation(view)
        info: dict[str, Any] = {
            "scenario": self.scenario,
            "opponent": self.opponent.name,
            "control_steps": steps,
        }
        if terminated or truncated:
            info["goal"] = goal
            info["is_success"] = goal == "home"
            info["episode_terms"] = dict(self.episode_terms)
        return obs, float(total), terminated, truncated, info

    def render(self):
        if self.render_mode != "rgb_array":
            return None
        if self._renderer is None:
            from hockey.visualization.renderer import SceneRenderer

            self._renderer = SceneRenderer(self.sim, width=640, height=360, camera="top")
        return self._renderer.render()

    def close(self) -> None:
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None
