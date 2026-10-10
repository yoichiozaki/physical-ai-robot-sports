"""Trace one serve of a learned policy and the receiver's decisions (why aces happen).

    python scripts/diagnostics/serve_trace.py [--service-line 1.62]

Prints every 3rd control step: time, the receiver's mode, its planned contact, the ball in the
receiver's frame and the receiver's gantry position.
"""

from __future__ import annotations

import argparse

import numpy as np

from tennis.control.factory import make_controller
from tennis.evaluation.conditions import CONDITIONS
from tennis.evaluation.match import Match
from tennis.sim.game import GameConfig
from tennis.sim.kinematics import world_to_player
from tennis.sim.physics import TennisSim


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--policy", default="checkpoints/ppo_league/league/gen_000.zip")
    p.add_argument("--service-line", type=float, default=None)
    p.add_argument("--seed", type=int, default=5)
    a = p.parse_args()
    sim = TennisSim()
    spec = f"hybrid:{a.policy}"
    server, receiver = (
        make_controller(spec, sim.cfg, "server"),
        make_controller(spec, sim.cfg, "receiver"),
    )
    rows = []

    def on_step(rec) -> None:
        if rec.referee.points:
            return
        info, st = rec.infos["away"], rec.state
        rows.append((round(st.time, 2), info.mode, info.contact,
                     np.round(world_to_player("away", st.ball_pos), 2), np.round(st.q["away"][:2], 2)))  # fmt: skip

    game = GameConfig(points=1, service_line=a.service_line)
    m = Match(
        sim, server, receiver, a.seed, CONDITIONS["nominal"], game, on_step, first_server="home"
    )
    m.run()
    pt = m.ref.points[0]
    print(f"point: server={pt.server} winner={pt.winner} reason={pt.reason} hits={pt.hits}")
    for t, mode, contact, ball, gantry in rows[::3]:
        c = "-" if contact is None else np.round(contact, 2)
        print(f"t={t:5.2f} mode={mode:8s} contact={c} ball={ball} gantry={gantry}")


if __name__ == "__main__":
    main()
