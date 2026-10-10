"""Identify the racket impact model from real swings (ball-machine drill).

Fits  v_out . n = a (v_in . n) + b s   and   v_out_t = c_t v_in_t  by least squares, where n
and s are the planned face normal and racket speed and v_in / v_out are measured.

    python scripts/calibrate_impact.py --hits 160   -> results/calibration.json["impact"]
    python scripts/calibrate_impact.py --serve      -> results/calibration.json["serve"]

The serve hits an almost stationary ball (the top of the toss bounce); its rebound is
stronger than the groundstroke model predicts, which gives PlannerConfig.slow_impact_gain.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from tennis.control.players import PhysicsPlayer, PlayerConfig
from tennis.evaluation.drills import run_drill
from tennis.sim.physics import TennisSim


def fit(records) -> dict:
    vin_n = np.array([r.v_in @ r.normal for r in records])
    s = np.array([r.racket_speed for r in records])
    vout_n = np.array([r.v_out @ r.normal for r in records])
    (a, b), *_ = np.linalg.lstsq(np.column_stack([vin_n, s]), vout_n, rcond=None)
    vt_in = np.array([r.v_in - (r.v_in @ r.normal) * r.normal for r in records])
    vt_out = np.array([r.v_out - (r.v_out @ r.normal) * r.normal for r in records])
    ct = float(np.sum(vt_in * vt_out) / np.sum(vt_in * vt_in))
    pred_n = a * vin_n + b * s
    return {
        "a": float(a),
        "b": float(b),
        "c_t": ct,
        "rms_normal_mps": float(np.sqrt(np.mean((pred_n - vout_n) ** 2))),
        "rms_tangential_mps": float(np.sqrt(np.mean(np.sum((ct * vt_in - vt_out) ** 2, axis=1)))),
        "mean_spin_radps": float(np.mean([np.linalg.norm(r.spin) for r in records])),
        "hits": len(records),
    }


def serve_gain(points: int = 24) -> dict:
    """Ratio of measured to modelled normal speed for serves planned *without* the gain."""
    from dataclasses import replace

    from tennis.control.players import opponent_factory
    from tennis.evaluation.match import Match
    from tennis.sim.game import GameConfig
    from tennis.sim.kinematics import world_to_player

    sim = TennisSim()
    home = PhysicsPlayer(sim.cfg, PlayerConfig("Physics"))
    home.planner.pc = replace(home.planner.pc, slow_impact_gain=1.0)
    away = opponent_factory(sim.cfg)["steady"]()
    ratios, pending = [], {}

    def on_step(rec) -> None:
        ref = rec.referee
        for e in rec.result.events:
            if e.kind == "racket_home" and ref.hits <= 1 and "t" not in pending:
                pending.update(t=e.time, plan=home.plan)
        if "t" in pending and not rec.sim._touch["racket_home"]:
            # same measurement as the groundstroke identification (drills): first sample after
            # the contact, gravity and drag undone back to the end of the contact
            t_hit, plan = pending.pop("t"), pending.pop("plan")
            if plan is not None and plan.v_out is not None and np.linalg.norm(plan.ball_vel) < 1.0:
                v = world_to_player("home", rec.sim.state().ball_vel)
                dt = rec.sim.time - (t_hit + 0.005)
                v = v + dt * (np.array([0.0, 0.0, 9.81]) + 0.0202 * np.linalg.norm(v) * v)
                ratios.append(float(v @ plan.normal) / float(plan.v_out @ plan.normal))

    game = GameConfig(points=points, max_point_time=3.0)
    Match(sim, home, away, 3, game=game, on_step=on_step, first_server="home").run()
    r = np.array(ratios)
    return {"slow_impact_gain": float(r.mean()), "std": float(r.std()), "serves": len(r)}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--hits", type=int, default=160)
    p.add_argument("--serve", action="store_true", help="identify the slow-impact (serve) gain")
    a = p.parse_args()
    if a.serve:
        path = Path("results/calibration.json")
        data = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
        data["serve"] = serve_gain()
        path.write_text(json.dumps(data, indent=2), encoding="utf-8")
        print(json.dumps(data["serve"], indent=2))
        return
    sim = TennisSim()
    player = PhysicsPlayer(sim.cfg, PlayerConfig("Physics"))
    recs = run_drill(sim, player, a.hits, seed=11)
    clean = [r for r in recs if r.clean]
    res = fit(clean)
    res["edge_hit_rate"] = 1.0 - len(clean) / max(len(recs), 1)
    path = Path("results/calibration.json")
    data = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    data["impact"] = res
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
