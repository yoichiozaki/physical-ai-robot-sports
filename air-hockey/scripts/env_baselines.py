"""Reference success rates of fixed (non-learned) shot choices in the training-evaluation setting.

Uses exactly the evaluation environment of the training runs (Stage 5, eval seed 100000):
the constant action "direct shot at the goal centre, 3.5 m/s" reproduces controller B.

    python scripts/env_baselines.py --episodes 240   -> results/env_baselines.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from hockey.learning.env import AirHockeyEnv, EnvConfig

ACTIONS = {
    "direct_center_3.5 (= B)": [1, 2, 1],
    "direct_center_4.5": [1, 2, 2],
    "direct_left_4.5": [1, 4, 2],
    "bank_left_4.5": [2, 2, 2],
    "bank_right_4.5": [3, 2, 2],
}


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--episodes", type=int, default=240)
    a = p.parse_args(argv)
    out = {}
    for name, act in ACTIONS.items():
        env = AirHockeyEnv(EnvConfig(action_mode="hybrid", stage=4))
        env.reset(seed=100000)
        succ = ep = 0
        while ep < a.episodes:
            _, _, term, trunc, info = env.step(np.array(act))
            if term or trunc:
                ep += 1
                succ += int(info["is_success"])
                env.reset()
        out[name] = succ / a.episodes
        print(f"{name:26s} {out[name]:.3f}", flush=True)
    Path("results/env_baselines.json").write_text(
        json.dumps({"episodes": a.episodes, "success": out}, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
