"""Key comparisons with confidence intervals -> results/summary.json and results/summary.md

    python scripts/summarize_results.py [--eval results/eval_main]

Paired comparisons use the same (opponent, condition, seed) cells, so differences are
estimated with a paired bootstrap over matches.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

COND_JA = {
    "nominal": "標準",
    "randomized": "ランダム化",
    "unknown_sticky": "未知(高摩擦)",
    "unknown_bouncy": "未知(高反発)",
}


def paired_diff(
    df: pd.DataFrame, a: str, b: str, metric: str, cond: str | None = None, n: int = 4000
) -> dict:
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


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--eval", default="results/eval_main")
    a = p.parse_args(argv)
    df = pd.read_csv(Path(a.eval) / "matches.csv")
    df["points"] = df["win"] + 0.5 * df["draw"]
    order = ["A", "B", "B+", "B*", "C", "D", "D*"]
    ctrls = sorted(
        df["controller"].unique(),
        key=lambda n: order.index(n.split(":")[0]) if n.split(":")[0] in order else 99,
    )
    table = []
    for c in ctrls:
        row = {"controller": c}
        for cond in COND_JA:
            g = df[(df["controller"] == c) & (df["condition"] == cond)]
            if g.empty:
                continue
            thr = g["threats"].sum()
            row[cond] = {
                "win": float(g["win"].mean()),
                "points": float(g["points"].mean()),
                "goals_for": float(g["goals_for"].mean()),
                "goals_against": float(g["goals_against"].mean()),
                "goal_diff": float(g["goal_diff"].mean()),
                "defense_success": float(g["saves"].sum() / thr) if thr else float("nan"),
                "prediction_error_m": float(g["prediction_error_m"].mean()),
                "latency_ms": float(g["latency_ms_mean"].mean()),
                "matches": len(g),
            }
        table.append(row)
    names = {c.split(":")[0]: c for c in ctrls}
    comparisons = {}
    pairs = [("D", "B"), ("D", "B+"), ("C", "B"), ("D*", "D"), ("B*", "B"), ("B", "A"), ("D", "C")]
    for x, y in pairs:
        if x in names and y in names:
            for metric in ("goal_diff", "goals_for", "points"):
                for cond in (None, *COND_JA):
                    key = f"{x}-{y}:{metric}:{cond or 'all'}"
                    comparisons[key] = paired_diff(df, names[x], names[y], metric, cond)
    out = {"by_controller": table, "paired_differences": comparisons}
    Path("results/summary.json").write_text(
        json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    # Markdown table for the README
    lines = [
        "| 制御器 | " + " | ".join(f"{v} 勝率 / 得失点差" for v in COND_JA.values()) + " |",
        "|---" * 5 + "|",
    ]
    for row in table:
        cells = []
        for cond in COND_JA:
            r = row.get(cond)
            cells.append("—" if r is None else f"{r['win'] * 100:.0f}% / {r['goal_diff']:+.2f}")
        lines.append(f"| {row['controller']} | " + " | ".join(cells) + " |")
    Path("results/summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    for k, v in comparisons.items():
        if ":all" in k or ":nominal" in k:
            print(f"{k:32s} {v['diff']:+.3f} [{v['lo']:+.3f}, {v['hi']:+.3f}] n={v['n']}")


if __name__ == "__main__":
    main()
