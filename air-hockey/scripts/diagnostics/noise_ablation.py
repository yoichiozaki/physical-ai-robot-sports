"""Why does the physics controller hit the puck less under the randomized condition?"""

import collections

import numpy as np

from hockey.controllers.opponents import make_opponent
from hockey.controllers.physics_based import PhysicsController
from hockey.evaluation.conditions import CONDITIONS, ConditionConfig
from hockey.evaluation.match import play_match
from hockey.observation import Geometry
from hockey.simulation import HockeySim
from hockey.simulation.game import GameConfig

sim = HockeySim()
geom = Geometry.from_sim_config(sim.cfg)
variants = {
    "nominal": CONDITIONS["nominal"],
    "randomized": CONDITIONS["randomized"],
    "rand_no_noise": ConditionConfig(
        "x", (0.05, 0.35), (0.75, 0.92), (0.6, 0.8), (0.8, 1.1), 0.0, 0.0
    ),
    "noise_only": ConditionConfig("y", obs_noise_pos=0.003, obs_noise_vel=0.03),
    "malletE_only": ConditionConfig("z", mallet_restitution=(0.6, 0.8)),
    "decay_only": ConditionConfig("w", puck_decay=(0.05, 0.35)),
}
for name, cond in variants.items():
    modes = collections.Counter()
    rows = []
    for seed in range(4):
        c = PhysicsController(geom)

        def on_step(rec, modes=modes):
            modes[rec.home_cmd.info.mode] += 1

        rows.append(
            play_match(
                sim,
                c,
                make_opponent("random", geom),
                600 + seed,
                cond,
                GameConfig(match_duration=60),
                on_step=on_step,
            ).summary()
        )
    f = lambda k: np.nanmean([x[k] for x in rows])  # noqa: E731
    tot = sum(modes.values())
    print(
        f"{name:14s} GF {f('goals_for'):.2f} hits {f('hits_home'):.1f} poss {f('possession_resets'):.1f} modes "
        + ", ".join(f"{k}:{v / tot:.2f}" for k, v in modes.most_common()),
        flush=True,
    )
