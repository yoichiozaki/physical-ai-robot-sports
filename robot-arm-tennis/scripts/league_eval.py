"""Evaluate the self-play league: round-robin matches, Elo ratings, tactic evolution.

    python scripts/league_eval.py --points 12 --matches 4 --workers 20

Players: every league snapshot (gen_000 = the Hybrid AI D the league started from, ...),
the physics baselines B / B+ / B* and the five scripted opponents. Every pair plays
``--matches`` matches (home/away and the first server alternate). Ratings are fitted on
the individual points with a Bradley-Terry model (Elo scale, gen_000 anchored at 1500)
that includes a serve advantage; 95% intervals by bootstrap over matches.

Outputs (results/league/): points.csv (one row per point), matches.csv, elo.json,
head_to_head.csv, tactics.csv (shot choices of every generation, from its home matches).
"""

from __future__ import annotations

import argparse
import itertools
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

LEAGUE = Path("checkpoints/ppo_league")
OUT = Path("results/league")
BASELINES = {"B: Physics": "physics", "B+: Physics+Rule": "physics_rule",
             "B*: Physics+Adapt": "physics_adaptive"}  # fmt: skip
SCRIPTED = ("steady", "random", "aggressive", "lob", "slow")
_CACHE: dict = {}


def players(league_dir: Path = LEAGUE) -> dict[str, str]:
    league = json.loads((league_dir / "league" / "league.json").read_text(encoding="utf-8"))
    # snapshots only (the final model is < 1k decisions after the last snapshot)
    out = {s["name"]: f"gen:{s['path']}" for s in league["snapshots"]}
    out.update(BASELINES)
    out.update({f"opp:{n}": f"opp:{n}" for n in SCRIPTED})
    return out


def _make(name: str, spec: str, cfg):
    from tennis.control.factory import make_controller
    from tennis.control.players import opponent_factory

    if spec.startswith("opp:"):
        return opponent_factory(cfg)[spec[4:]]()
    if spec.startswith("gen:"):
        return make_controller("hybrid:" + spec[4:], cfg, name)
    return make_controller(spec, cfg, name)


def play(task: dict) -> dict:
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    import torch

    torch.set_num_threads(1)
    from tennis.evaluation.conditions import CONDITIONS
    from tennis.evaluation.match import Match
    from tennis.sim.game import GameConfig
    from tennis.sim.physics import TennisSim

    if "sim" not in _CACHE:
        _CACHE["sim"] = TennisSim()
    sim = _CACHE["sim"]
    home = _make(task["home"], task["home_spec"], sim.cfg)
    away = _make(task["away"], task["away_spec"], sim.cfg)
    game = GameConfig(points=task["points"], service_line=task.get("service_line"))
    m = Match(
        sim, home, away, task["seed"], CONDITIONS["nominal"], game, first_server=task["first"]
    )
    res = m.run()
    pts = [{"home": task["home"], "away": task["away"], "seed": task["seed"], "winner": p.winner,
            "server": p.server, "reason": p.reason, "hits": p.hits} for p in m.ref.points]  # fmt: skip
    # every rally shot that landed in: where (receiver's frame) and whether it came back in
    shots, log = [], m.ref.shot_log
    for i, s in enumerate(log):
        if s.get("serve") or s["landing"] != "in" or s["pos"] is None:
            continue
        hitter = s["player"]
        receiver = "away" if hitter == "home" else "home"
        x, y = s["pos"]
        if receiver == "away":  # the away frame is the world rotated by 180 degrees
            x, y = -x, -y
        nxt = log[i + 1] if i + 1 < len(log) else None
        back = (
            nxt is not None and nxt["player"] == receiver and nxt["landing"] in ("in", "volleyed")
        )
        shots.append({"hitter": task[hitter], "receiver": task[receiver], "x": x, "y": y,
                      "returned": int(back)})  # fmt: skip
    return {"points": pts, "strategy": res.strategy_counts, "home": task["home"],
            "away": task["away"], "seed": task["seed"], "shots": shots}  # fmt: skip


def fit_elo(df: pd.DataFrame, names: list[str], anchor: str, l2: float = 1e-4) -> dict:
    """Bradley-Terry on points: P(server wins) = sigmoid(c (R_server - R_receiver) + s)."""
    from scipy.optimize import minimize

    idx = {n: i for i, n in enumerate(names)}
    srv = np.where(df["server"] == "home", df["home"], df["away"])
    rcv = np.where(df["server"] == "home", df["away"], df["home"])
    i_s = np.array([idx[n] for n in srv])
    i_r = np.array([idx[n] for n in rcv])
    y = (np.where(df["winner"] == "home", df["home"], df["away"]) == srv).astype(float)
    c = np.log(10.0) / 400.0
    n = len(names)

    def nll(theta):
        r, s = theta[:n], theta[n]
        z = c * (r[i_s] - r[i_r]) + s
        p = 1.0 / (1.0 + np.exp(-z))
        loss = -np.sum(y * np.log(p + 1e-12) + (1 - y) * np.log(1 - p + 1e-12)) + l2 * np.sum(r**2)
        g = p - y
        grad_r = np.zeros(n)
        np.add.at(grad_r, i_s, c * g)
        np.add.at(grad_r, i_r, -c * g)
        return loss, np.concatenate([grad_r + 2 * l2 * r, [np.sum(g)]])

    sol = minimize(nll, np.zeros(n + 1), jac=True, method="L-BFGS-B")
    r = sol.x[:n] - sol.x[idx[anchor]] + 1500.0
    return {"ratings": dict(zip(names, r.tolist(), strict=True)), "serve_logit": float(sol.x[n])}


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--points", type=int, default=12)
    p.add_argument("--matches", type=int, default=4)
    p.add_argument("--workers", type=int, default=os.cpu_count() or 4)
    p.add_argument("--seed0", type=int, default=40_000)
    p.add_argument("--bootstrap", type=int, default=200)
    p.add_argument("--out", default=str(OUT))
    p.add_argument("--league", default=str(LEAGUE), help="league run directory")
    p.add_argument("--service-line", type=float, default=None, help="tennis service-box rule")
    a = p.parse_args(argv)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    roster = players(Path(a.league))
    names = list(roster)
    tasks = []
    for (i, x), (j, y) in itertools.combinations(enumerate(names), 2):
        for k in range(a.matches):
            home, away = (x, y) if k % 2 == 0 else (y, x)
            tasks.append({"home": home, "away": away, "home_spec": roster[home],
                          "away_spec": roster[away], "seed": a.seed0 + 1000 * i + 37 * j + k,
                          "first": "home" if (k // 2) % 2 == 0 else "away", "points": a.points,
                          "service_line": a.service_line})  # fmt: skip
    t0 = time.time()
    results = []
    with ProcessPoolExecutor(a.workers) as ex:
        futs = [ex.submit(play, t) for t in tasks]
        for n_done, f in enumerate(as_completed(futs), 1):
            results.append(f.result())
            if n_done % max(1, len(tasks) // 20) == 0:
                print(
                    f"[league-eval] {n_done}/{len(tasks)} matches ({time.time() - t0:.0f}s)",
                    flush=True,
                )
    pts = pd.DataFrame([pt for r in results for pt in r["points"]])
    pts.to_csv(out / "points.csv", index=False)
    pd.DataFrame([s for r in results for s in r["shots"]]).to_csv(out / "shots.csv", index=False)
    matches = pd.DataFrame([{"home": r["home"], "away": r["away"], "seed": r["seed"],
                             "home_points": sum(pt["winner"] == "home" for pt in r["points"]),
                             "away_points": sum(pt["winner"] == "away" for pt in r["points"])}
                            for r in results])  # fmt: skip
    matches.to_csv(out / "matches.csv", index=False)
    fit = fit_elo(pts, names, "gen_000")
    rng = np.random.default_rng(0)
    keys = list(matches[["home", "away", "seed"]].itertuples(index=False, name=None))
    by_match = {k: g for k, g in pts.groupby(["home", "away", "seed"])}
    boots = []
    for _ in range(a.bootstrap):
        pick = [keys[i] for i in rng.integers(len(keys), size=len(keys))]
        boots.append(fit_elo(pd.concat([by_match[k] for k in pick]), names, "gen_000")["ratings"])
    ci = {n: [float(np.percentile([b[n] for b in boots], 2.5)),
              float(np.percentile([b[n] for b in boots], 97.5))] for n in names}  # fmt: skip
    league = json.loads((Path(a.league) / "league" / "league.json").read_text(encoding="utf-8"))
    steps = {s["name"]: s["timesteps"] for s in league["snapshots"]}
    elo = {"ratings": fit["ratings"], "ci95": ci, "serve_logit": fit["serve_logit"],
           "timesteps": steps, "points_per_match": a.points, "matches_per_pair": a.matches,
           "matches": len(matches), "points": len(pts), "wall_time_s": round(time.time() - t0, 1)}  # fmt: skip
    (out / "elo.json").write_text(json.dumps(elo, indent=2), encoding="utf-8")
    # head-to-head point-win rate (row vs column)
    wins = pd.DataFrame(0.0, index=names, columns=names)
    tot = pd.DataFrame(0.0, index=names, columns=names)
    for (x, y), g in pts.groupby(["home", "away"]):
        hx = float((g["winner"] == "home").sum())
        wins.loc[x, y] += hx
        wins.loc[y, x] += len(g) - hx
        tot.loc[x, y] += len(g)
        tot.loc[y, x] += len(g)
    (wins / tot.replace(0, np.nan)).to_csv(out / "head_to_head.csv")
    # shot choices of every learned player (from the matches it played at home)
    rows = []
    for r in results:
        for label, cnt in r["strategy"].items():
            rows.append({"player": r["home"], "label": label, "count": cnt})
    pd.DataFrame(rows).to_csv(out / "tactics.csv", index=False)
    order = sorted(names, key=lambda n: -fit["ratings"][n])
    for n in order:
        print(f"{n:20s} {fit['ratings'][n]:7.1f}  [{ci[n][0]:.0f}, {ci[n][1]:.0f}]")


if __name__ == "__main__":
    main()
