"""Training scenarios and the curriculum (Stage 1-5 train, Stage 6 = held-out evaluation)."""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from hockey.simulation.physics import HockeySim

SCENARIOS = ("strike", "straight", "bank", "match")


@dataclass(frozen=True)
class StageConfig:
    name: str = "1-strike"
    scenarios: dict[str, float] = field(default_factory=lambda: {"strike": 1.0})
    opponents: tuple[str, ...] = ("stationary",)
    condition: str = "nominal"
    drill_time: float = 5.0
    match_time: float = 15.0
    launch_speed: tuple[float, float] = (1.5, 3.5)


DEFAULT_STAGES: tuple[StageConfig, ...] = (
    StageConfig("1-strike", {"strike": 1.0}),
    StageConfig("2-straight", {"straight": 0.6, "strike": 0.4}),
    StageConfig("3-bank", {"bank": 0.4, "straight": 0.3, "strike": 0.3}, launch_speed=(1.5, 4.0)),
    StageConfig(
        "4-simple-opponent",
        {"match": 0.6, "bank": 0.15, "straight": 0.15, "strike": 0.1},
        ("stationary", "random", "reactive"),
        condition="randomized",
        launch_speed=(1.5, 4.0),
    ),
    StageConfig(
        "5-multi-opponent",
        {"match": 0.8, "bank": 0.08, "straight": 0.07, "strike": 0.05},
        ("stationary", "random", "reactive", "defensive", "aggressive"),
        condition="randomized",
        launch_speed=(1.5, 4.5),
    ),
)


def sample_scenario(stage: StageConfig, rng: np.random.Generator) -> str:
    names = list(stage.scenarios)
    w = np.array([stage.scenarios[n] for n in names], dtype=float)
    return names[int(rng.choice(len(names), p=w / w.sum()))]


def reset_scenario(
    sim: HockeySim, scenario: str, stage: StageConfig, rng: np.random.Generator
) -> str:
    """Place puck and robots for ``scenario``; returns the side holding the puck."""
    cfg = sim.cfg
    L2, W2 = cfg.table.length / 2, cfg.table.width / 2
    rp = cfg.puck.radius
    g2 = cfg.table.goal_width / 2
    ylim = W2 - rp
    home = sim.home_position("home") + np.array([0.0, rng.uniform(-0.15, 0.15)])
    if scenario == "strike":
        p = (rng.uniform(-0.75, -0.15), rng.uniform(-0.40, 0.40))
        sp, ang = rng.uniform(0.0, 0.3), rng.uniform(-math.pi, math.pi)
        sim.reset(puck_pos=p, puck_vel=(sp * math.cos(ang), sp * math.sin(ang)), home_pos=home)
        return "home"
    if scenario in ("straight", "bank"):
        p0 = np.array([rng.uniform(0.05, 0.6), rng.uniform(-0.40, 0.40)])
        y_t = rng.uniform(-(g2 - rp), g2 - rp)
        if scenario == "bank":
            side = 1.0 if rng.random() < 0.5 else -1.0
            y_t = side * 2 * ylim - y_t
        d = np.array([-L2, y_t]) - p0
        d /= np.linalg.norm(d)
        speed = rng.uniform(*stage.launch_speed)
        sim.reset(puck_pos=p0, puck_vel=d * speed, home_pos=home)
        return "away"
    if scenario == "match":
        side = "home" if rng.random() < 0.5 else "away"
        sx = -1.0 if side == "home" else 1.0
        sim.reset(puck_pos=(sx * 0.45, rng.uniform(-0.2, 0.2)), puck_vel=(0.0, 0.0), home_pos=home)
        return side
    raise ValueError(f"Unknown scenario {scenario}")
