"""Experiment runner: controllers x opponents x conditions x seeds, in parallel.

Writes per-match rows (CSV) and aggregated statistics (mean, std, 95% CI).
"""

from __future__ import annotations

import json
import math
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from tennis.config import to_dict


@dataclass(frozen=True)
class ControllerEntry:
    name: str
    spec: str


@dataclass(frozen=True)
class EvalExperimentConfig:
    output: str = "results/eval_main"
    controllers: tuple[ControllerEntry, ...] = (
        ControllerEntry("A: Naive", "naive"),
        ControllerEntry("B: Physics", "physics"),
    )
    opponents: tuple[str, ...] = ("steady", "random", "aggressive", "lob", "slow")
    conditions: tuple[str, ...] = ("nominal",)
    seeds: tuple[int, ...] = tuple(range(20_000, 20_010))
    points: int = 20
    workers: int = 0  # 0 -> os.cpu_count()
    notes: dict[str, str] = field(default_factory=dict)


_CACHE: dict[str, Any] = {}


def _worker_init() -> None:
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    try:
        import torch

        torch.set_num_threads(1)
    except ImportError:
        pass


def run_one(task: dict[str, Any]) -> dict[str, Any]:
    """Play one match (executed in a worker process)."""
    from tennis.control.factory import make_controller
    from tennis.control.players import opponent_factory
    from tennis.evaluation.conditions import CONDITIONS
    from tennis.evaluation.match import play_match
    from tennis.sim.game import GameConfig
    from tennis.sim.physics import TennisSim

    if "sim" not in _CACHE:
        _CACHE["sim"] = TennisSim()
        _CACHE["opp"] = opponent_factory(_CACHE["sim"].cfg)
    sim = _CACHE["sim"]
    key = f"ctrl:{task['spec']}"
    if key not in _CACHE:
        _CACHE[key] = make_controller(task["spec"], sim.cfg, task["controller"])
    home = _CACHE[key]
    away = _CACHE["opp"][task["opponent"]]()
    t0 = time.perf_counter()
    res = play_match(
        sim,
        home,
        away,
        task["seed"],
        CONDITIONS[task["condition"]],
        GameConfig(points=task["points"]),
        first_server="home" if task["seed"] % 2 == 0 else "away",
    )
    extra = {}
    ma = getattr(home, "model_adapter", None)
    if ma is not None and ma._base is not None:
        extra = {
            "id_e": ma.e, "id_mu": ma.mu, "id_k_drag": ma.k_drag,
            "id_wind_x": float(ma.wind[0]), "id_wind_y": float(ma.wind[1]),
            "id_shifted": int(ma.shifted), "id_bounces": ma.n_bounces, "id_air": ma.n_air,
        }  # fmt: skip
    ia = getattr(home, "impact_adapter", None)
    if ia is not None:
        extra["id_impact_k"] = ia.k
    return {
        "controller": task["controller"],
        "opponent": task["opponent"],
        "condition": task["condition"],
        "seed": task["seed"],
        "wall_time_s": time.perf_counter() - t0,
        **res.summary(),
        **extra,
    }


METRICS = (
    "point_win_rate",
    "win",
    "draw",
    "loss",
    "points_won",
    "points_lost",
    "avg_rally_hits",
    "in_rate",
    "net_rate",
    "out_rate",
    "serve_in_rate",
    "return_rate",
    "landing_error_m",
    "prediction_error_m",
    "latency_ms_mean",
    "latency_ms_p95",
    "winners",
    "unforced_errors",
)


def aggregate(df: pd.DataFrame, by: list[str]) -> pd.DataFrame:
    rows = []
    for keys, g in df.groupby(by, sort=False):
        keys = keys if isinstance(keys, tuple) else (keys,)
        row: dict[str, Any] = dict(zip(by, keys, strict=True))
        row["matches"] = len(g)
        for m in METRICS:
            if m not in g:
                continue
            vals = g[m].astype(float).dropna()
            n = len(vals)
            mean = float(vals.mean()) if n else float("nan")
            std = float(vals.std(ddof=1)) if n > 1 else float("nan")
            row[f"{m}_mean"] = mean
            row[f"{m}_std"] = std
            row[f"{m}_ci95"] = 1.96 * std / math.sqrt(n) if n > 1 else float("nan")
        opp = g["opportunities"].sum()
        row["return_rate_pooled"] = float(g["returns_in"].sum() / opp) if opp else float("nan")
        shots = g["shots"].sum()
        row["in_rate_pooled"] = float((g["in_rate"] * g["shots"]).sum() / shots) if shots else 0.0
        rows.append(row)
    return pd.DataFrame(rows)


def write_summaries(df: pd.DataFrame, out: Path, meta: dict) -> None:
    out.mkdir(parents=True, exist_ok=True)
    df = df.sort_values(["controller", "opponent", "condition", "seed"])
    df.to_csv(out / "matches.csv", index=False)
    aggregate(df, ["controller", "condition", "opponent"]).to_csv(
        out / "summary_by_opponent.csv", index=False
    )
    by_cond = aggregate(df, ["controller", "condition"])
    by_cond.to_csv(out / "summary_by_condition.csv", index=False)
    (out / "summary_by_condition.json").write_text(
        by_cond.to_json(orient="records", indent=2, force_ascii=False), encoding="utf-8"
    )
    (out / "meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")


def merge(sources: list[Path], out: Path) -> pd.DataFrame:
    """Combine several experiment outputs (e.g. baselines + learned controllers)."""
    frames, metas = [], []
    for src in sources:
        frames.append(pd.read_csv(src / "matches.csv"))
        metas.append(json.loads((src / "meta.json").read_text(encoding="utf-8")))
    df = pd.concat(frames, ignore_index=True)
    meta = {
        "merged_from": [str(s) for s in sources],
        "matches": len(df),
        "wall_time_s": round(sum(m["wall_time_s"] for m in metas), 1),
        "parts": metas,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    write_summaries(df, out, meta)
    return df


def run_experiment(cfg: EvalExperimentConfig, verbose: bool = True) -> pd.DataFrame:
    out = Path(cfg.output)
    out.mkdir(parents=True, exist_ok=True)
    tasks = [
        {
            "controller": c.name,
            "spec": c.spec,
            "opponent": o,
            "condition": cond,
            "seed": s,
            "points": cfg.points,
        }
        for c in cfg.controllers
        for o in cfg.opponents
        for cond in cfg.conditions
        for s in cfg.seeds
    ]
    workers = cfg.workers or os.cpu_count() or 1
    t0 = time.time()
    rows: list[dict[str, Any]] = []
    with ProcessPoolExecutor(max_workers=workers, initializer=_worker_init) as ex:
        futures = [ex.submit(run_one, t) for t in tasks]
        for i, fut in enumerate(as_completed(futures), 1):
            rows.append(fut.result())
            if verbose and (i % max(1, len(tasks) // 20) == 0 or i == len(tasks)):
                print(f"[eval] {i}/{len(tasks)} matches ({time.time() - t0:.0f}s)", flush=True)
    df = pd.DataFrame(rows)
    meta = {
        "config": to_dict(cfg),
        "matches": len(df),
        "wall_time_s": round(time.time() - t0, 1),
        "workers": workers,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    write_summaries(df, out, meta)
    return df


def paired_diff(df: pd.DataFrame, a: str, b: str, metric: str, cond: str | None = None,
                n: int = 4000) -> dict:  # fmt: skip
    """Mean of (a - b) over the same (opponent, condition, seed) cells, 95% bootstrap CI."""
    sel = df if cond is None else df[df["condition"] == cond]
    keys = ["opponent", "condition", "seed"]
    pa = sel[sel["controller"] == a].set_index(keys)[metric]
    pb = sel[sel["controller"] == b].set_index(keys)[metric]
    common = pa.index.intersection(pb.index)
    d = (pa.loc[common] - pb.loc[common]).to_numpy(dtype=float)
    d = d[~np.isnan(d)]
    if len(d) == 0:
        return {"diff": float("nan"), "lo": float("nan"), "hi": float("nan"), "n": 0}
    rng = np.random.default_rng(0)
    boots = rng.choice(d, (n, len(d))).mean(axis=1)
    return {
        "diff": float(d.mean()),
        "lo": float(np.percentile(boots, 2.5)),
        "hi": float(np.percentile(boots, 97.5)),
        "n": len(d),
    }
