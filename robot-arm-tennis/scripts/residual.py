"""Collect ball-machine hits for residual learning (parallel), then train / evaluate the model.

    python scripts/residual.py collect --hits 6000 --condition randomized
    python scripts/residual.py train
    python scripts/residual.py evaluate --hits 300

Outputs: results/residual/hits_*.npz, checkpoints/residual/model.pt(+.json),
results/residual/train.json, results/residual/drill_eval.json
"""

from __future__ import annotations

import argparse
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

OUT = Path("results/residual")
MODEL = Path("checkpoints/residual/model.pt")
COURT_X, COURT_Y = 3.0 + 0.02, 1.2 + 0.02


def _collect(args: tuple[int, int, str, str | None, bool, bool]) -> dict:
    n, seed, cond, model_path, online, resample = args
    from tennis.control.players import PhysicsPlayer, PlayerConfig
    from tennis.evaluation.conditions import CONDITIONS
    from tennis.evaluation.drills import run_drill
    from tennis.learning.residual import AimCorrector, ResidualModel
    from tennis.sim.physics import TennisSim

    sim = TennisSim()
    player = PhysicsPlayer(sim.cfg, PlayerConfig("Physics"))
    residual = ResidualModel.load(model_path) if model_path is not None else None
    if online:  # online system identification (+ residual gated by change detection)
        from tennis.learning.adaptation import make_adaptive

        make_adaptive(player, residual)
    elif residual is not None:
        player.aim_correction = AimCorrector(residual)
    recs = run_drill(sim, player, n, seed=seed, condition=CONDITIONS[cond], resample=resample)
    rows = []
    for r in recs:
        landed = r.landing is not None and r.landing[0] > 0
        rows.append(
            {
                "features": r.features,
                "aim": np.array(r.features[6:8]),
                "target": np.array(r.intended),
                "landing": None if not landed else np.array(r.landing[:2]),
                "net": bool(r.net),
                "clean": bool(r.clean),
            }
        )
    return {"rows": rows}


def collect(hits: int, cond: str, seed0: int, workers: int, model: str | None = None,
            online: bool = False, resample: bool = True, sessions: int = 16):  # fmt: skip
    """``sessions`` drills of hits/sessions balls each (one sampled environment per session
    when ``resample`` is False), run on ``workers`` processes."""
    per = max(1, hits // sessions)
    tasks = [(per, seed0 + i, cond, model, online, resample) for i in range(sessions)]
    with ProcessPoolExecutor(workers) as ex:
        parts = list(ex.map(_collect, tasks))
    return [r for p in parts for r in p["rows"]]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("action", choices=["collect", "train", "evaluate"])
    p.add_argument("--hits", type=int, default=6000)
    p.add_argument("--condition", default="randomized")
    p.add_argument("--workers", type=int, default=min(16, os.cpu_count() or 4))
    p.add_argument("--seed", type=int, default=1000)
    p.add_argument("--tag", default=None, help="output name suffix (collect)")
    p.add_argument("--data", nargs="+", default=["hits_nominal.npz"], help="training files (train)")
    a = p.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    if a.action == "collect":
        t0 = time.time()
        rows = [
            r for r in collect(a.hits, a.condition, a.seed, a.workers) if r["landing"] is not None
        ]
        X = np.stack([r["features"] for r in rows])
        Y = np.stack([r["landing"] - r["aim"] for r in rows])
        np.savez(OUT / f"hits_{a.tag or a.condition}.npz", X=X, Y=Y)
        print(
            f"collected {len(rows)} landed hits in {time.time() - t0:.0f}s; "
            f"mean |err| {np.linalg.norm(Y, axis=1).mean():.3f} m"
        )
    elif a.action == "train":
        from tennis.learning.residual import ResidualModel

        parts = [np.load(OUT / f) for f in a.data]
        X = np.concatenate([d["X"] for d in parts])
        Y = np.concatenate([d["Y"] for d in parts])
        rng = np.random.default_rng(0)
        idx = rng.permutation(len(X))
        n_val = len(X) // 5
        val, tr = idx[:n_val], idx[n_val:]
        t0 = time.time()
        model, hist = ResidualModel.fit(X[tr], Y[tr], epochs=120, weight_decay=1e-3)
        pred = model.predict(X[val])
        res = {
            "data": a.data,
            "train_samples": len(tr),
            "val_samples": len(val),
            "val_error_without_model_m": float(np.linalg.norm(Y[val], axis=1).mean()),
            "val_error_with_model_m": float(np.linalg.norm(Y[val] - pred, axis=1).mean()),
            "val_median_without_m": float(np.median(np.linalg.norm(Y[val], axis=1))),
            "val_median_with_m": float(np.median(np.linalg.norm(Y[val] - pred, axis=1))),
            "train_seconds": round(time.time() - t0, 1),
            "loss_history": hist[::20],
        }
        model.save(MODEL)
        (OUT / "train.json").write_text(json.dumps(res, indent=2), encoding="utf-8")
        print(json.dumps({k: v for k, v in res.items() if k != "loss_history"}, indent=2))
    else:
        out = {}
        variants = (
            ("physics", None, False),
            ("physics+residual", str(MODEL), False),
            ("physics+online-id", None, True),
            ("physics+residual+online-id", str(MODEL), True),
        )
        for cond in ("nominal", "randomized", "unknown_wind", "unknown_court"):
            out[cond] = {}
            for name, model, online in variants:
                rows = collect(a.hits, cond, 50000, a.workers, model, online, resample=False)
                landed = [r for r in rows if r["landing"] is not None]
                err = [float(np.linalg.norm(r["landing"] - r["target"])) for r in landed]
                inside = [
                    abs(r["landing"][0]) <= COURT_X and abs(r["landing"][1]) <= COURT_Y
                    for r in landed
                ]
                out[cond][name] = {
                    "hits": len(rows),
                    "landing_error_mean_m": float(np.mean(err)),
                    "landing_error_median_m": float(np.median(err)),
                    "in_court_rate": float(np.sum(inside) / max(len(rows), 1)),
                    "net_rate": float(np.mean([r["net"] for r in rows])),
                }
                print(cond, name, out[cond][name], flush=True)
        (OUT / "drill_eval.json").write_text(json.dumps(out, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
