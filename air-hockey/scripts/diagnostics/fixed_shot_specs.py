"""Do fixed shot specs differ in effectiveness against each opponent? (sanity check for learning)"""

import sys
import time

from hockey.controllers.opponents import make_opponent
from hockey.controllers.physics_based import PhysicsController
from hockey.controllers.planning import ShotSpec, Strategy
from hockey.evaluation.match import play_match
from hockey.observation import Geometry
from hockey.simulation import HockeySim
from hockey.simulation.game import GameConfig

sim = HockeySim()
geom = Geometry.from_sim_config(sim.cfg)
specs = {
    "direct_c_3.5": ShotSpec(Strategy.DIRECT, 0.0, 3.5),
    "direct_+.09_4.5": ShotSpec(Strategy.DIRECT, 0.09, 4.5),
    "direct_-.09_4.5": ShotSpec(Strategy.DIRECT, -0.09, 4.5),
    "bankL_0_4.5": ShotSpec(Strategy.BANK_LEFT, 0.0, 4.5),
    "bankR_0_4.5": ShotSpec(Strategy.BANK_RIGHT, 0.0, 4.5),
}
opps = sys.argv[1:] or ["stationary", "defensive", "aggressive", "physics"]
for opp in opps:
    for name, spec in specs.items():
        t = time.time()
        gf = ga = 0
        for seed in range(3):
            ctrl = PhysicsController(geom)
            ctrl.forced_spec = spec
            r = play_match(
                sim, ctrl, make_opponent(opp, geom), seed, game=GameConfig(match_duration=60)
            ).summary()
            gf += r["goals_for"]
            ga += r["goals_against"]
        print(
            f"{opp:10s} {name:16s} GF {gf / 3:.1f} GA {ga / 3:.1f} [{time.time() - t:.1f}s]",
            flush=True,
        )
