"""Learned controllers: SB3 PPO policies wrapped as ``Controller`` objects.

They reproduce the environment's observation/decision pipeline exactly (same encoder,
same possession clock, same action adapter and decision interval) so that a policy
trained in ``AirHockeyEnv`` can play full matches.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from hockey.controllers.adaptation import OnlinePuckEstimator
from hockey.controllers.base import Command
from hockey.controllers.prediction import PuckModel
from hockey.learning.actions import (
    DEFAULT_AIMS,
    DEFAULT_SPEEDS,
    DirectTargetAction,
    HybridStrategyAction,
)
from hockey.observation import Geometry, PlayerView, SideClock, encode_observation


def _run_env_config(model_path: Path) -> dict:
    for candidate in (model_path.parent / "config.json", model_path.parent.parent / "config.json"):
        if candidate.exists():
            return json.loads(candidate.read_text(encoding="utf-8")).get("env", {})
    return {}


class PolicyController:
    def __init__(
        self,
        model_path: str | Path,
        geom: Geometry,
        model: PuckModel | None = None,
        name: str | None = None,
        deterministic: bool = True,
        adaptive: bool = False,
    ) -> None:
        import torch
        from stable_baselines3 import PPO

        torch.set_num_threads(1)
        path = Path(model_path)
        env_cfg = _run_env_config(path)
        self.mode = env_cfg.get("action_mode", "direct")
        self.action_repeat = int(env_cfg.get("action_repeat", 2))
        self.min_steps = int(env_cfg.get("decision_min_steps", 5))
        self.max_steps = int(env_cfg.get("decision_max_steps", 100))
        self.geom = geom
        self.policy = PPO.load(str(path), device="cpu")
        if self.mode == "hybrid":
            self.adapter: DirectTargetAction | HybridStrategyAction = HybridStrategyAction(
                geom,
                model or PuckModel(),
                tuple(env_cfg.get("hybrid_aims", DEFAULT_AIMS)),
                tuple(env_cfg.get("hybrid_speeds", DEFAULT_SPEEDS)),
            )
            self.name = name or "Hybrid AI"
            if adaptive:
                ex = self.adapter.executor
                ex.estimator = OnlinePuckEstimator(geom, ex.prior_model)
        else:
            self.adapter = DirectTargetAction(geom)
            self.name = name or "PPO (pure RL)"
        self.deterministic = deterministic
        self.clock = SideClock()
        self._k = 0
        self._since = 0
        self.last_action: np.ndarray | None = None

    def reset(self, rng: np.random.Generator) -> None:
        self.adapter.reset(rng)
        self.clock.reset()
        self._k = 0
        self._since = 0

    def set_speed_scale(self, scale: float) -> None:
        if isinstance(self.adapter, HybridStrategyAction):
            self.adapter.executor.set_speed_scale(scale)

    def _decide_now(self, view: PlayerView) -> bool:
        """Same decision schedule as ``AirHockeyEnv.step``."""
        if self._k == 0:
            return True
        if self.mode == "hybrid":
            return self._since >= self.max_steps or (
                self._since >= self.min_steps and self.adapter.needs_decision(view)
            )
        return self._since >= self.action_repeat

    def act(self, view: PlayerView) -> Command:
        self.clock.update(view)
        if self._decide_now(view):
            obs = encode_observation(view, self.geom, self.clock.value)
            if self.adapter.feature_dim:
                obs = np.concatenate([obs, self.adapter.features(view)]).astype(np.float32)
            action, _ = self.policy.predict(obs, deterministic=self.deterministic)
            self.adapter.set_action(action)
            self.last_action = np.asarray(action)
            self._since = 0
        self._k += 1
        self._since += 1
        return self.adapter.command(view)
