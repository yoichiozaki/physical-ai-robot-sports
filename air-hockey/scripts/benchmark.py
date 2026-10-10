"""Physics and controller micro-benchmarks (Phase 2/3 evidence) -> results/benchmark.json

* contact restitution calibration (target vs measured, several speeds/angles)
* analytic prediction vs MuJoCo free flight (incl. wall bounces), several horizons
* strike accuracy of the physics controller (direction error, outgoing speed) and the
  defence success against straight/bank shots for Reactive vs Physics
* decision latency and simulation throughput

    python scripts/benchmark.py
"""

from __future__ import annotations

import json
import math
import time
from pathlib import Path

import numpy as np

from hockey.controllers.base import apply_command
from hockey.controllers.physics_based import PhysicsController
from hockey.controllers.prediction import PuckModel, PuckPredictor
from hockey.controllers.reactive import ReactiveController
from hockey.observation import Geometry, player_view
from hockey.simulation import HockeySim, PhysicsParams


def restitution_calibration() -> list[dict]:
    rows = []
    for target in (0.6, 0.75, 0.85, 0.95):
        sim = HockeySim()
        sim.set_params(PhysicsParams(0.15, target, 0.7))
        vals = []
        for speed in (0.5, 1.0, 2.0, 4.0, 6.0):
            for ang in (90.0, 60.0, 30.0):
                a = math.radians(ang)
                sim.reset(puck_pos=(0.3, 0.2), puck_vel=(speed * math.cos(a), speed * math.sin(a)))
                vin = None
                for _ in range(200):
                    r = sim.step()
                    for e in r.events:
                        if e.kind == "wall" and vin is None:
                            vin = e.puck_vel_before[1]
                    if vin is not None and not sim._in_contact["wall"]:
                        vals.append(-sim.state().puck_vel[1] / vin)
                        break
        rows.append({"target": target, "mean": float(np.mean(vals)), "std": float(np.std(vals)),
                     "min": float(np.min(vals)), "max": float(np.max(vals)), "n": len(vals)})  # fmt: skip
    return rows


def prediction_accuracy(n: int = 200) -> dict:
    sim = HockeySim()
    geom = Geometry.from_sim_config(sim.cfg)
    pred = PuckPredictor(geom, PuckModel(decay=0.15, wall_restitution=0.85))
    rng = np.random.default_rng(0)
    horizons = (0.25, 0.5, 1.0)
    errs: dict[float, list[float]] = {h: [] for h in horizons}
    for _ in range(n):
        p0 = np.array([rng.uniform(-0.5, 0.5), rng.uniform(-0.35, 0.35)])
        ang = rng.uniform(-math.pi, math.pi)
        sp = rng.uniform(0.5, 4.0)
        v0 = np.array([sp * math.cos(ang), sp * math.sin(ang)])
        sim.reset(puck_pos=p0, puck_vel=v0, home_pos=(-0.95, 0.45), away_pos=(0.95, -0.45))
        traj = pred.predict(p0, v0, horizon=1.0)
        t = 0.0
        for h in horizons:
            ok = True
            while t < h - 1e-9:
                r = sim.step()
                t += sim.control_dt
                if r.goal or r.mallet_contact:
                    ok = False
                    break
            if not ok:
                break
            x, y, _, _ = traj.state_at(h)
            errs[h].append(math.dist((x, y), sim.state().puck_pos))
    return {f"{h}s": {"mean_m": float(np.mean(e)), "p95_m": float(np.percentile(e, 95)), "n": len(e)}
            for h, e in errs.items()}  # fmt: skip


def strike_accuracy(n: int = 60) -> dict:
    sim = HockeySim()
    geom = Geometry.from_sim_config(sim.cfg)
    rng = np.random.default_rng(1)
    errs, speeds, icpt, struck = [], [], [], 0
    for trial in range(n):
        if trial % 2:
            p0 = (rng.uniform(0.2, 0.6), rng.uniform(-0.35, 0.35))
            d = np.array([-1.0, rng.uniform(-0.3, 0.3)]) - np.array(p0)
            v0 = tuple(d / np.linalg.norm(d) * rng.uniform(1.0, 3.0))
        else:
            p0, v0 = (rng.uniform(-0.75, -0.2), rng.uniform(-0.35, 0.35)), (0.0, 0.0)
        sim.reset(puck_pos=p0, puck_vel=v0)
        ctrl = PhysicsController(geom)
        ctrl.reset(np.random.default_rng(0))
        plan, hit_t = None, None
        for _ in range(150):
            cmd = ctrl.act(player_view(sim.state(), "home"))
            if cmd.info.mode == "strike" and ctrl._strike is not None:
                plan = ctrl._strike
            apply_command(sim, "home", cmd)
            r = sim.step()
            for e in r.events:
                if e.kind == "home" and hit_t is None:
                    hit_t = sim.time
                    if plan is not None:
                        icpt.append(math.dist(plan.puck_contact, e.puck_pos))
            if (hit_t is not None and sim.time - hit_t > 0.06) or r.goal:
                break
        if hit_t is None or plan is None:
            continue
        struck += 1
        v = sim.state().puck_vel
        a, b = math.atan2(plan.direction[1], plan.direction[0]), math.atan2(v[1], v[0])
        errs.append(math.degrees(abs((b - a + math.pi) % (2 * math.pi) - math.pi)))
        speeds.append(float(np.hypot(*v)))
    return {
        "trials": n, "struck": struck,
        "direction_error_deg_mean": float(np.mean(errs)), "direction_error_deg_median": float(np.median(errs)),
        "direction_error_deg_p90": float(np.percentile(errs, 90)),
        "intercept_error_m_mean": float(np.mean(icpt)), "out_speed_mean": float(np.mean(speeds)),
    }  # fmt: skip


def defense_drill(n: int = 100) -> dict:
    sim = HockeySim()
    geom = Geometry.from_sim_config(sim.cfg)
    out = {}
    for name, mk in (
        ("reactive", lambda: ReactiveController(geom)),
        ("physics", lambda: PhysicsController(geom)),
    ):
        rng = np.random.default_rng(2)
        saved = {"straight": [0, 0], "bank": [0, 0]}
        for i in range(n):
            kind = "bank" if i % 2 else "straight"
            p0 = np.array([rng.uniform(0.0, 0.6), rng.uniform(-0.4, 0.4)])
            ylim = 0.5 - 0.032
            gy = rng.uniform(-0.1, 0.1)
            if kind == "bank":
                gy = (1 if rng.random() < 0.5 else -1) * 2 * ylim - gy
            d = np.array([-1.0, gy]) - p0
            v0 = d / np.linalg.norm(d) * rng.uniform(2.0, 4.5)
            sim.reset(puck_pos=p0, puck_vel=v0)
            ctrl = mk()
            ctrl.reset(np.random.default_rng(0))
            goal = None
            for _ in range(100):
                apply_command(sim, "home", ctrl.act(player_view(sim.state(), "home")))
                r = sim.step()
                if r.goal:
                    goal = r.goal
                    break
            saved[kind][0] += int(goal != "away")
            saved[kind][1] += 1
        out[name] = {k: v[0] / v[1] for k, v in saved.items()}
    return out


def throughput() -> dict:
    sim = HockeySim()
    geom = Geometry.from_sim_config(sim.cfg)
    ctrl = PhysicsController(geom)
    sim.reset(puck_pos=(0.3, 0.1), puck_vel=(-2.0, 0.5))
    lat = []
    t0 = time.perf_counter()
    for _ in range(1500):
        v = player_view(sim.state(), "home")
        t = time.perf_counter()
        cmd = ctrl.act(v)
        lat.append((time.perf_counter() - t) * 1e3)
        apply_command(sim, "home", cmd)
        if sim.step().goal:
            sim.reset(puck_pos=(0.3, 0.1), puck_vel=(-2.0, 0.5))
    dt = time.perf_counter() - t0
    return {"physics_controller_latency_ms_mean": float(np.mean(lat)),
            "physics_controller_latency_ms_p99": float(np.percentile(lat, 99)),
            "realtime_factor_with_controller": 1500 * sim.control_dt / dt}  # fmt: skip


def main() -> None:
    t = time.time()
    report = {
        "restitution_calibration": restitution_calibration(),
        "prediction_accuracy": prediction_accuracy(),
        "strike_accuracy": strike_accuracy(),
        "defense_drill_success": defense_drill(),
        "throughput": throughput(),
    }
    report["wall_time_s"] = round(time.time() - t, 1)
    Path("results").mkdir(exist_ok=True)
    Path("results/benchmark.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
