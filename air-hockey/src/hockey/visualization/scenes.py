"""Scripted initial situations for demo clips (applied after the match face-off)."""

from __future__ import annotations

import math
from collections.abc import Callable

from hockey.simulation.physics import HockeySim

SCENES: dict[str, tuple[tuple[float, float], tuple[float, float] | None]] = {
    # name: (puck position, puck velocity or None = aim at the home goal)
    "strike": ((-0.45, 0.12), (0.0, 0.0)),
    "straight_defense": ((0.55, 0.25), None),
    "bank_defense": ((0.6, -0.3), None),
}


def scene_setup(scene: str, speed: float = 3.2) -> Callable[[HockeySim], None] | None:
    if scene == "match":
        return None
    pos, vel = SCENES[scene]

    def setup(sim: HockeySim) -> None:
        v = vel
        if v is None:  # aim at the home goal, directly or off the +y side wall
            L2 = sim.cfg.table.length / 2
            ylim = sim.cfg.table.width / 2 - sim.cfg.puck.radius
            gy = 0.03 if scene == "straight_defense" else 2 * ylim - 0.03
            dx, dy = -L2 - pos[0], gy - pos[1]
            n = math.hypot(dx, dy)
            v = (speed * dx / n, speed * dy / n)
        sim.reset(puck_pos=pos, puck_vel=v, keep_time=False)

    return setup
