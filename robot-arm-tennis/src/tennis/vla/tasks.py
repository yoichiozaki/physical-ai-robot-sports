"""Instructed return drills: a ball machine feeds the home robot, a language instruction says
where the return should land. Some instructions refer to the opponent robot's position, so
they can only be followed by looking at the scene (vision-language grounding).

Frames: home player frame (= world). The opponent half is x > 0; seen from the home robot,
"left" is +y and "right" is -y.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from tennis.evaluation.drills import random_feed

DEPTH_X = {"deep": 2.35, "short": 1.25}
DEPTH_RANGE = {"deep": (1.8, 3.02), "short": (0.45, 1.8)}
SIDE_Y = {"left": 0.75, "middle": 0.0, "right": -0.75}
SIDE_RANGE = {"left": (0.35, 1.22), "middle": (-0.35, 0.35), "right": (-1.22, -0.35)}
COURT_X, COURT_Y = 3.02, 1.22  # singles court incl. lines (opponent half: 0 < x <= COURT_X)

EXPLICIT = {
    ("deep", "left"): "hit a deep shot to the left",
    ("deep", "middle"): "hit a deep shot down the middle",
    ("deep", "right"): "hit a deep shot to the right",
    ("short", "left"): "hit a short shot to the left",
    ("short", "middle"): "hit a short shot down the middle",
    ("short", "right"): "hit a short shot to the right",
}
RELATIONAL = {
    "away": "hit the ball away from the opponent",
    "at": "hit the ball straight at the opponent",
}
_SIDE_JA = {"left": "左", "middle": "センター", "right": "右"}
_DEPTH_JA = {"deep": "深く", "short": "短く"}
INSTRUCTION_JA = {
    **{v: f"{_DEPTH_JA[d]}・{_SIDE_JA[s]}へ" for (d, s), v in EXPLICIT.items()},
    RELATIONAL["away"]: "相手のいない方へ",
    RELATIONAL["at"]: "相手の正面へ",
}


@dataclass(frozen=True)
class Task:
    world: str
    seed: int
    instruction: str
    kind: str  # "explicit" | "away" | "at"
    zone: tuple[str, str]  # (depth, side) the return must land in
    target: tuple[float, float]  # aim used by the physics expert (zone centre + jitter)
    opp_y: float  # opponent carriage position, world y
    feed_p0: tuple[float, float, float]
    feed_v0: tuple[float, float, float]


def opponent_gantry_y(opp_y_world: float) -> float:
    """The away robot's frame is the world rotated by 180 degrees."""
    return -opp_y_world


def sample_task(world: str, gravity: float, seed: int, p_relational: float = 0.4) -> Task:
    rng = np.random.default_rng(seed)
    # opponent stands clearly left or right of the centre line (its rail spans y in [-1.9, 0.9])
    opp_y = float(rng.uniform(0.45, 0.9) if rng.random() < 0.4 else rng.uniform(-1.6, -0.45))
    if rng.random() < p_relational:
        kind = "away" if rng.random() < 0.5 else "at"
        opp_side = "left" if opp_y > 0 else "right"
        if kind == "away":
            zone = ("deep", "right" if opp_side == "left" else "left")
        else:
            zone = ("deep", opp_side)
        instruction = RELATIONAL[kind]
    else:
        kind = "explicit"
        zone = (str(rng.choice(["deep", "short"])), str(rng.choice(["left", "middle", "right"])))
        instruction = EXPLICIT[zone]
    jitter = rng.normal(0.0, 0.08, 2)
    target = (DEPTH_X[zone[0]] + float(jitter[0]), SIDE_Y[zone[1]] + float(jitter[1]))
    p0, v0 = random_feed(rng, gravity)
    return Task(world, seed, instruction, kind, zone, target, opp_y,
                tuple(float(v) for v in p0), tuple(float(v) for v in v0))  # fmt: skip


def in_court(landing: np.ndarray | None) -> bool:
    return landing is not None and 0.0 < landing[0] <= COURT_X and abs(landing[1]) <= COURT_Y


def in_zone(landing: np.ndarray | None, zone: tuple[str, str], margin: float = 0.1) -> bool:
    if not in_court(landing):
        return False
    (x0, x1), (y0, y1) = DEPTH_RANGE[zone[0]], SIDE_RANGE[zone[1]]
    return x0 - margin <= landing[0] <= x1 + margin and y0 - margin <= landing[1] <= y1 + margin
