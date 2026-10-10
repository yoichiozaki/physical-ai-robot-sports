"""Controller factory used by the evaluation runner, the video and the tests.

Spec strings:
    naive                  A   no ball physics for returns (straight-line extrapolation)
    physics                B   physics prediction + shot solver, always aims at the centre
    physics_rule           B+  B with a hand-written open-court tactic
    physics_adaptive       B*  B+ with online system identification + gated residual model
    direct:<ckpt>          C   PPO outputs racket pitch / yaw / speed (no ballistic shot model)
    hybrid:<ckpt>          D   PPO chooses the shot, physics layer with online identification
    hybrid_static:<ckpt>   D-  ablation: the same policy without online identification
"""

from __future__ import annotations

from pathlib import Path

from tennis.control.players import (
    NaivePlayer,
    PhysicsPlayer,
    PlayerConfig,
    centre_strategy,
    rule_strategy,
)
from tennis.sim.config import TennisConfig

RESIDUAL = "checkpoints/residual/model.pt"


def _residual():
    if not Path(RESIDUAL).exists():
        return None
    from tennis.learning.residual import ResidualModel

    return ResidualModel.load(RESIDUAL)


def make_controller(spec: str, cfg: TennisConfig, name: str | None = None) -> PhysicsPlayer:
    kind, _, ckpt = spec.partition(":")
    if kind == "naive":
        return NaivePlayer(cfg, PlayerConfig(name or "A: Naive"))
    if kind == "physics":
        return PhysicsPlayer(cfg, PlayerConfig(name or "B: Physics"), centre_strategy())
    if kind == "physics_rule":
        return PhysicsPlayer(cfg, PlayerConfig(name or "B+: Physics+Rule"), rule_strategy())
    if kind == "physics_adaptive":
        from tennis.learning.adaptation import make_adaptive

        player = PhysicsPlayer(cfg, PlayerConfig(name or "B*: Physics+Adapt"), rule_strategy())
        return make_adaptive(player, _residual())
    if kind in ("direct", "hybrid", "hybrid_static"):
        from tennis.learning.env import make_agent_player
        from tennis.learning.policy import PolicyStrategy

        default = {"direct": "C: PPO direct", "hybrid": "D: Hybrid AI", "hybrid_static": "D-"}
        player = make_agent_player(cfg, kind != "hybrid_static", RESIDUAL, name or default[kind])
        mode = "direct" if kind == "direct" else "hybrid"
        player.strategy = PolicyStrategy(ckpt, mode, player)
        return player
    raise ValueError(f"unknown controller spec: {spec}")
