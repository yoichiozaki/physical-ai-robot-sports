"""Deploy a trained PPO policy as the shot-selection strategy of a physics player."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from tennis.control.players import PhysicsPlayer, PlayerView
from tennis.learning.env import decode_action, shot_obs


class PolicyStrategy:
    """``strategy(view, pred, rng)`` backed by a Stable-Baselines3 policy (deterministic)."""

    def __init__(self, checkpoint: str | Path, mode: str, player: PhysicsPlayer) -> None:
        from stable_baselines3 import PPO

        self.model = PPO.load(str(checkpoint), device="cpu")
        self.mode = mode
        self.player = player
        self.last_action: np.ndarray | None = None

    def __call__(self, view: PlayerView, pred, rng: np.random.Generator):
        obs = shot_obs(view, self.player)
        action, _ = self.model.predict(obs, deterministic=True)
        self.last_action = np.asarray(action)
        return decode_action(self.mode, action)
