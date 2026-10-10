"""Identify the ball model used by the controllers from the MuJoCo simulation.

Measures (no robots involved except the racket test):
  * free flight without spin  -> drag constant k_d
  * free flight with spin     -> Magnus constant k_m (and spin decay)
  * court bounces             -> e_z, friction mu, grip factor beta
  * racket rebound            -> effective normal restitution of a servo-held racket

    python scripts/calibrate.py   -> results/calibration.json
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
from scipy.optimize import minimize

from tennis.control.ballistics import BallModel, BallPredictor
from tennis.sim.physics import TennisSim
from tennis.sim.trajectory import JointTrajectory

FAR = {
    "home": np.array([0.0, 1.9, 0.0, 1.2, -2.4, 0.0, 0.0, 0.0]),
    "away": np.array([0.0, 1.9, 0.0, 1.2, -2.4, 0.0, 0.0, 0.0]),
}


def fly(sim: TennisSim, p0, v0, spin, T: float, dt: float = 0.02):
    """Free flight samples (robots parked out of the way)."""
    sim.reset(ball_pos=p0, ball_vel=v0, ball_spin=spin, robot_q=FAR)
    out = []
    n = int(T / sim.control_dt)
    for _ in range(n):
        r = sim.step()
        if r.events or r.lost:
            break
        s = sim.state()
        out.append((s.time, s.ball_pos.copy(), s.ball_spin.copy()))
    return out


def flight_error(model: BallModel, data) -> float:
    pred = BallPredictor(model, dt=0.002)
    err = 0.0
    for p0, v0, spin, samples in data:
        tr = pred.predict(
            np.array(p0),
            np.array(v0),
            horizon=samples[-1][0] + 0.01,
            spin=np.array(spin),
            max_bounces=0,
        )
        for t, pos, _ in samples:
            pp, _ = tr.state_at(t)
            err += float(np.sum((pp - pos) ** 2))
    return err


def main() -> None:
    sim = TennisSim()
    rng = np.random.default_rng(0)
    out: dict = {}
    # 1) drag (no spin), high launches that stay in the air ~0.8 s
    data = []
    for _ in range(12):
        sp = rng.uniform(4.0, 11.0)
        el = rng.uniform(0.25, 0.7)
        az = rng.uniform(-0.4, 0.4)
        v0 = sp * np.array([math.cos(el) * math.cos(az), math.cos(el) * math.sin(az), math.sin(el)])
        p0 = (-2.5, 0.0, 1.0)
        data.append((p0, tuple(v0), (0.0, 0.0, 0.0), fly(sim, p0, v0, (0, 0, 0), 0.8)))
    res = minimize(
        lambda k: flight_error(BallModel(k_drag=float(k[0]), k_magnus=0.0), data),
        x0=[0.02],
        method="Nelder-Mead",
    )
    k_d = float(res.x[0])
    rms_nodrag = math.sqrt(
        flight_error(BallModel(k_drag=0.0, k_magnus=0.0), data) / sum(len(d[3]) for d in data)
    )
    rms_drag = math.sqrt(res.fun / sum(len(d[3]) for d in data))
    out["drag"] = {
        "k_drag": k_d,
        "rms_position_error_m": rms_drag,
        "rms_if_drag_ignored_m": rms_nodrag,
    }
    # 2) Magnus (top/back/side spin), spin decay
    data = []
    decay = []
    for _ in range(12):
        sp = rng.uniform(5.0, 10.0)
        el = rng.uniform(0.3, 0.6)
        v0 = sp * np.array([math.cos(el), 0.0, math.sin(el)])
        spin = rng.uniform(-150, 150, 3) * np.array([0.3, 1.0, 0.5])
        p0 = (-2.5, 0.0, 1.0)
        samples = fly(sim, p0, v0, spin, 0.7)
        data.append((p0, tuple(v0), tuple(spin), samples))
        s0, s1 = np.linalg.norm(spin), np.linalg.norm(samples[-1][2])
        decay.append((samples[-1][0], s0, s1))
    res = minimize(
        lambda k: flight_error(BallModel(k_drag=k_d, k_magnus=float(k[0])), data),
        x0=[0.003],
        method="Nelder-Mead",
    )
    k_m = float(res.x[0])
    rms_nomag = math.sqrt(
        flight_error(BallModel(k_drag=k_d, k_magnus=0.0), data) / sum(len(d[3]) for d in data)
    )
    out["magnus"] = {
        "k_magnus": k_m,
        "rms_position_error_m": math.sqrt(res.fun / sum(len(d[3]) for d in data)),
        "rms_if_spin_ignored_m": rms_nomag,
        "spin_retained_after_0.7s": float(np.mean([s1 / s0 for _, s0, s1 in decay])),
    }
    # 3) court bounces (no spin): incoming speed/angle sweep
    rows = []
    for _ in range(40):
        vz = -rng.uniform(1.5, 6.0)
        vt = rng.uniform(0.0, 8.0)
        sim.reset(ball_pos=(1.0, 0.0, 0.2), ball_vel=(vt, 0.0, vz), robot_q=FAR)
        vin = None
        for _ in range(40):
            r = sim.step()
            ev = [e for e in r.events if e.kind == "floor"]
            if ev and vin is None:
                vin = ev[0].ball_vel
            if vin is not None and not sim._touch["floor"]:
                _, v = sim.ball()
                rows.append((vin[0], vin[2], v[0], v[2]))
                break
    rows = np.array(rows)
    e_z = float(np.mean(-rows[:, 3] / rows[:, 1]))

    def bounce_err(x):
        mu, beta = x
        loss = np.minimum(mu * (1 + e_z) * np.abs(rows[:, 1]), (1 - beta) * np.abs(rows[:, 0]))
        pred = np.sign(rows[:, 0]) * (np.abs(rows[:, 0]) - loss)
        return float(np.sum((pred - rows[:, 2]) ** 2))

    res = minimize(bounce_err, x0=[0.5, 0.6], method="Nelder-Mead")
    out["bounce"] = {
        "e_z": e_z,
        "e_z_std": float(np.std(-rows[:, 3] / rows[:, 1])),
        "mu": float(res.x[0]),
        "beta": float(res.x[1]),
        "rms_tangential_error_mps": math.sqrt(res.fun / len(rows)),
    }
    # 4) racket rebound: home robot holds the racket face up (normal +z), balls dropped on it
    p_up = sim.kin.shoulder(0.6, 0.4) + np.array([0.6, -0.5, 0.0])
    q_up = sim.kin.ik(p_up, np.array([0.0, 0.0, 1.0]), (0.6, 0.4))
    ratios = []
    for _ in range(10):
        sim.reset(robot_q={"home": q_up, "away": FAR["away"]})
        sim.set_trajectory("home", JointTrajectory.hold(q_up))
        for _ in range(20):
            sim.step()
        s = sim.state()
        top = s.racket_pos["home"] + s.racket_normal["home"] * (0.05 + rng.uniform(0.25, 0.6))
        speed = rng.uniform(2.0, 5.0)
        sim.place_ball(top, -s.racket_normal["home"] * speed)
        vin = None
        for _ in range(30):
            r = sim.step()
            ev = [e for e in r.events if e.kind == "racket_home"]
            if ev and vin is None:
                vin = ev[0].ball_vel
            if vin is not None and not sim._touch["racket_home"]:
                _, v = sim.ball()
                n = sim.state().racket_normal["home"]
                ratios.append(-float(v @ n) / float(vin @ n))
                break
    out["racket"] = {
        "e_normal": float(np.mean(ratios)),
        "e_normal_std": float(np.std(ratios)),
        "n": len(ratios),
    }
    out["model"] = {
        "k_drag": k_d,
        "k_magnus": k_m,
        "bounce_e": e_z,
        "bounce_mu": out["bounce"]["mu"],
        "bounce_beta": out["bounce"]["beta"],
        "racket_e": out["racket"]["e_normal"],
    }
    Path("results").mkdir(exist_ok=True)
    Path("results/calibration.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
