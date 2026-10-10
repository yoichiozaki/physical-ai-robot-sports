"""Experiment runner: controllers x opponents x conditions x seeds, in parallel.

Writes per-match rows (CSV/JSON) and aggregated statistics (mean, std, 95% CI).
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

from hockey.config import to_dict


@dataclass(frozen=True)
class ControllerEntry:
    name: str
    spec: str


@dataclass(frozen=True)
class EvalExperimentConfig:
    output: str = "results/eval_main"
    controllers: tuple[ControllerEntry, ...] = (
        ControllerEntry("Reactive", "reactive"),
        ControllerEntry("Physics", "physics"),
    )
    opponents: tuple[str, ...] = ("stationary", "random", "reactive", "defensive", "aggressive")
    conditions: tuple[str, ...] = ("nominal",)
    seeds: tuple[int, ...] = tuple(range(10_000, 10_010))
    match_duration: float = 60.0
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
    from hockey.controllers.factory import make_controller
    from hockey.controllers.opponents import make_opponent
    from hockey.evaluation.conditions import CONDITIONS
    from hockey.evaluation.match import play_match
    from hockey.observation import Geometry
    from hockey.simulation import HockeySim
    from hockey.simulation.game import GameConfig

    if "sim" not in _CACHE:
        _CACHE["sim"] = HockeySim()
        _CACHE["geom"] = Geometry.from_sim_config(_CACHE["sim"].cfg)
    sim, geom = _CACHE["sim"], _CACHE["geom"]
    key = f"ctrl:{task['spec']}"
    if key not in _CACHE:
        _CACHE[key] = make_controller(task["spec"], geom)
    home = _CACHE[key]
    away = make_opponent(task["opponent"], geom)
    t0 = time.perf_counter()
    res = play_match(
        sim,
        home,
        away,
        task["seed"],
        CONDITIONS[task["condition"]],
        GameConfig(match_duration=task["duration"]),
    )
    row = {
        "controller": task["controller"],
        "opponent": task["opponent"],
        "condition": task["condition"],
        "seed": task["seed"],
        "wall_time_s": time.perf_counter() - t0,
        **res.summary(),
    }
    return row


METRICS = (
    "win",
    "draw",
    "loss",
    "goals_for",
    "goals_against",
    "goal_diff",
    "defense_success_rate",
    "avg_rally_hits",
    "prediction_error_m",
    "intercept_error_m",
    "latency_ms_mean",
    "latency_ms_p95",
    "own_goals",
    "anomalies",
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
        # pooled defence success (threat-weighted)
        thr = g["threats"].sum()
        row["defense_success_pooled"] = float(g["saves"].sum() / thr) if thr else float("nan")
        row["points_rate"] = float((g["win"] + 0.5 * g["draw"]).mean())
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
            "duration": cfg.match_duration,
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


def bootstrap_diff(
    a: np.ndarray, b: np.ndarray, n: int = 5000, seed: int = 0
) -> tuple[float, float, float]:
    """Mean difference a-b with a 95% bootstrap confidence interval."""
    rng = np.random.default_rng(seed)
    diffs = rng.choice(a, (n, len(a))).mean(1) - rng.choice(b, (n, len(b))).mean(1)
    return (
        float(a.mean() - b.mean()),
        float(np.percentile(diffs, 2.5)),
        float(np.percentile(diffs, 97.5)),
    )
