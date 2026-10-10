"""Quick comparison: physics vs physics_adaptive under several conditions."""

import sys
import time

import numpy as np

from hockey.controllers.factory import make_controller
from hockey.controllers.opponents import make_opponent
from hockey.evaluation.conditions import CONDITIONS
from hockey.evaluation.match import play_match
from hockey.observation import Geometry
from hockey.simulation import HockeySim
from hockey.simulation.game import GameConfig

sim = HockeySim()
geom = Geometry.from_sim_config(sim.cfg)
conds = sys.argv[1:] or ["nominal", "unknown_sticky", "unknown_bouncy"]
for cond in conds:
    for spec in ("physics", "physics_adaptive"):
        t = time.time()
        rows = []
        for seed in range(4):
            ctrl = make_controller(spec, geom)
            r = play_match(
                sim,
                ctrl,
                make_opponent("aggressive", geom),
                seed,
                CONDITIONS[cond],
                GameConfig(match_duration=60),
            )
            s = r.summary()
            if hasattr(ctrl, "estimator") and ctrl.estimator is not None:
                m = ctrl.estimator.model
                s["est"] = (
                    f"c={m.decay:.3f} e={m.wall_restitution:.3f} seg={ctrl.estimator.n_segments} b={ctrl.estimator.n_bounces}"
                )
            rows.append(s)
        f = lambda k: np.nanmean([x[k] for x in rows])  # noqa: E731
        print(
            f"{cond:15s} {spec:17s} GF {f('goals_for'):.2f} GA {f('goals_against'):.2f} def {f('defense_success_rate'):.2f} "
            f"pred {f('prediction_error_m'):.4f} icpt {f('intercept_error_m'):.4f} [{time.time() - t:.0f}s] {rows[-1].get('est', '')}",
            flush=True,
        )
