"""Controller interface shared by baselines, learned policies and scripted opponents."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

import numpy as np

from hockey.observation import PlayerView


@dataclass
class DecisionInfo:
    """What a controller decided and why (player frame). Used for HUD/overlays/metrics.

    Only real algorithm outputs are stored here; nothing is fabricated for display.
    """

    mode: str = ""  # e.g. "defend", "position", "strike", "retreat", "track"
    strategy: str = ""  # e.g. "Direct shot", "Bank shot (L)", "Defense"
    predicted_path: list[tuple[float, float]] = field(default_factory=list)
    bounce_points: list[tuple[float, float]] = field(default_factory=list)
    intercept: tuple[float, float] | None = None  # predicted puck centre at impact/block
    intercept_time: float | None = None  # seconds from now
    mallet_contact: tuple[float, float] | None = None
    shot_path: list[tuple[float, float]] = field(default_factory=list)
    extra: dict[str, float] = field(default_factory=dict)


@dataclass
class Command:
    target: np.ndarray  # desired mallet centre (player frame)
    info: DecisionInfo = field(default_factory=DecisionInfo)
    # Optional time-stamped follow-up target, executed by the 1 kHz low-level loop
    # ``switch_delay`` seconds from now (precise strike launch timing).
    switch_target: np.ndarray | None = None
    switch_delay: float = 0.0


def apply_command(sim, player: str, cmd: Command) -> None:
    """Send a player-frame command to the simulator (world frame)."""
    from hockey.observation import to_world

    if cmd.switch_target is not None:
        sim.set_target(
            player,
            to_world(player, cmd.target),
            to_world(player, cmd.switch_target),
            sim.time + max(cmd.switch_delay, 0.0),
        )
    else:
        sim.set_target(player, to_world(player, cmd.target))


@runtime_checkable
class Controller(Protocol):
    name: str

    def reset(self, rng: np.random.Generator) -> None: ...

    def act(self, view: PlayerView) -> Command: ...
