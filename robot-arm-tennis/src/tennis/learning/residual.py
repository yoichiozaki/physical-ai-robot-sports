"""Learned correction of the physics shot model (residual learning) + online bias adaptation.

The physics planner predicts where a shot lands; reality differs (racket friction and the
spin it creates, Magnus curvature, arm compliance, tracking error). A small MLP learns the
landing error  e = landing - aim  from the robot's own drill hits, using only quantities known
at planning time (planned contact state, aim, planned racket normal/speed, outgoing velocity).
At play time the aim is iterated as  aim <- target - e_hat(aim)  (one step per re-plan).

On top, ``OnlineBias`` tracks the mean residual of the robot's own recent shots during a
match (exponential moving average) and subtracts it, adapting to unknown conditions such as
wind or a heavier ball without retraining.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import torch
from torch import nn

from tennis.control.planner import HitPlan, ShotSpec

FEATURE_NAMES = (
    "ball_x", "ball_y", "ball_z", "vin_x", "vin_y", "vin_z", "aim_x", "aim_y", "speed",
    "vout_x", "vout_y", "vout_z", "n_x", "n_y", "n_z", "racket_speed",
)  # fmt: skip


def plan_features(plan: HitPlan) -> np.ndarray:
    shot = plan.shot
    aim = shot.target if isinstance(shot, ShotSpec) else (0.0, 0.0)
    speed = shot.speed if isinstance(shot, ShotSpec) else 0.0
    v_out = plan.v_out if plan.v_out is not None else np.zeros(3)
    return np.concatenate(
        [plan.ball_pos, plan.ball_vel, aim, [speed], v_out, plan.normal, [plan.racket_speed]]
    ).astype(np.float32)


class _MLP(nn.Module):
    def __init__(self, n_in: int, hidden: int = 64) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_in, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
            nn.Linear(hidden, 2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ResidualModel:
    def __init__(self, mean: np.ndarray, std: np.ndarray, net: _MLP) -> None:
        self.mean, self.std, self.net = mean, std, net.eval()

    @classmethod
    def fit(cls, X: np.ndarray, Y: np.ndarray, epochs: int = 400, seed: int = 0, lr: float = 3e-3,
            weight_decay: float = 1e-4) -> tuple[ResidualModel, list[float]]:  # fmt: skip
        torch.manual_seed(seed)
        mean, std = X.mean(0), X.std(0) + 1e-6
        net = _MLP(X.shape[1])
        opt = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=weight_decay)
        xt = torch.tensor((X - mean) / std, dtype=torch.float32)
        yt = torch.tensor(Y, dtype=torch.float32)
        hist = []
        for _ in range(epochs):
            perm = torch.randperm(len(xt))
            for i in range(0, len(xt), 256):
                idx = perm[i : i + 256]
                loss = nn.functional.smooth_l1_loss(net(xt[idx]), yt[idx], beta=0.1)
                opt.zero_grad()
                loss.backward()
                opt.step()
            hist.append(float(loss.detach()))
        return cls(mean, std, net), hist

    def predict(self, X: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            x = torch.tensor((np.atleast_2d(X) - self.mean) / self.std, dtype=torch.float32)
            return self.net(x).numpy()

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.net.state_dict(), path)
        meta = {
            "mean": self.mean.tolist(),
            "std": self.std.tolist(),
            "features": list(FEATURE_NAMES),
        }
        path.with_suffix(".json").write_text(json.dumps(meta), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> ResidualModel:
        path = Path(path)
        meta = json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))
        net = _MLP(len(meta["mean"]))
        net.load_state_dict(torch.load(path, map_location="cpu"))
        return cls(np.array(meta["mean"]), np.array(meta["std"]), net)


@dataclass
class OnlineBias:
    """EMA of the remaining landing error of own shots (adapts to unknown conditions)."""

    rate: float = 0.25
    clip: float = 0.6
    value: np.ndarray | None = None
    n: int = 0

    def reset(self) -> None:
        self.value, self.n = None, 0

    def update(self, residual: np.ndarray) -> None:
        r = np.clip(np.asarray(residual, float), -self.clip, self.clip)
        self.value = r if self.value is None else (1 - self.rate) * self.value + self.rate * r
        self.n += 1


class AimCorrector:
    """Callable used by PhysicsPlayer.aim_correction: target -> corrected aim."""

    def __init__(
        self, model: ResidualModel | None = None, online: bool = False, gain: float = 1.0
    ) -> None:
        self.model = model
        self.bias = OnlineBias() if online else None
        self.gain = gain
        self.gate: Callable[[], bool] | None = None  # False -> residual switched off
        self.target: tuple[float, float] | None = None
        self.aim: np.ndarray | None = None

    def reset(self) -> None:
        self.target, self.aim = None, None
        if self.bias is not None:
            self.bias.reset()

    def __call__(self, plan: HitPlan | None, shot: ShotSpec) -> ShotSpec:
        target = np.array(shot.target, dtype=float)
        active = self.model is not None and (self.gate is None or self.gate())
        # the model was learned on groundstrokes; slow contacts (the serve) are out of range
        slow = plan is not None and float(np.linalg.norm(plan.ball_vel)) < 3.0
        if self.target != shot.target or plan is None or not active or slow:
            self.target, self.aim = shot.target, target.copy()
        elif isinstance(plan.shot, ShotSpec):
            err = self.model.predict(plan_features(plan))[0]
            self.aim = target - self.gain * err  # fixed-point iteration over re-plans
        aim = self.aim.copy()
        if self.bias is not None and self.bias.value is not None:
            aim = aim - self.bias.value
        aim[0] = float(np.clip(aim[0], 0.5, 3.4))
        aim[1] = float(np.clip(aim[1], -1.5, 1.5))
        return replace(shot, target=(float(aim[0]), float(aim[1])))

    def observe(self, plan: HitPlan, landing: np.ndarray) -> None:
        """Feedback after one of our shots landed (player frame)."""
        if self.bias is None or not isinstance(plan.shot, ShotSpec):
            return
        aim = np.array(plan.shot.target)
        pred_err = (
            self.model.predict(plan_features(plan))[0] if self.model is not None else np.zeros(2)
        )
        self.bias.update(np.asarray(landing[:2]) - aim - pred_err)
