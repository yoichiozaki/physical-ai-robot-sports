"""Head-to-head matches between two players (point-win rate of A with a bootstrap 95% CI).

    python scripts/head_to_head.py --a A=gen:<a.zip> --b B=gen:<b.zip> --matches 40 \
        --out results/league/a_vs_b.json [--service-line 1.62]

Player specs as in scripts/league_eval.py: gen:<policy.zip>, physics | physics_rule |
physics_adaptive, opp:<scripted name>.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from league_eval import play


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--a", required=True, help="NAME=SPEC")
    p.add_argument("--b", required=True, help="NAME=SPEC")
    p.add_argument("--matches", type=int, default=40)
    p.add_argument("--points", type=int, default=12)
    p.add_argument("--seed0", type=int, default=60_000)
    p.add_argument("--workers", type=int, default=os.cpu_count() or 4)
    p.add_argument("--out", required=True)
    p.add_argument("--service-line", type=float, default=None)
    a = p.parse_args(argv)
    (na, sa), (nb, sb) = a.a.split("=", 1), a.b.split("=", 1)
    tasks = []
    for k in range(a.matches):
        home, away = ((na, sa), (nb, sb)) if k % 2 == 0 else ((nb, sb), (na, sa))
        tasks.append({"home": home[0], "away": away[0], "home_spec": home[1], "away_spec": away[1],
                      "seed": a.seed0 + k, "first": "home" if (k // 2) % 2 == 0 else "away",
                      "points": a.points, "service_line": a.service_line})  # fmt: skip
    t0 = time.time()
    with ProcessPoolExecutor(a.workers) as ex:
        results = list(ex.map(play, tasks))
    won = []  # per match: points won by A, points played
    for r in results:
        a_side = "home" if r["home"] == na else "away"
        pts = r["points"]
        won.append((sum(pt["winner"] == a_side for pt in pts), len(pts)))
    w = np.array(won, dtype=float)
    rate = w[:, 0].sum() / w[:, 1].sum()
    rng = np.random.default_rng(0)
    boots = [w[i, 0].sum() / w[i, 1].sum() for i in rng.integers(len(w), size=(2000, len(w)))]
    out = {"a": na, "b": nb, "a_spec": sa, "b_spec": sb, "matches": len(w),
           "points": int(w[:, 1].sum()), "a_point_win_rate": float(rate),
           "ci95": [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))],
           "wall_time_s": round(time.time() - t0, 1)}  # fmt: skip
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
