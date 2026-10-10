"""Summarize the VLA evaluations -> results/vla/summary.{csv,json} + figures.

    python scripts/vla/summarize_vla.py

Inputs: results/vla/<run>/episodes.jsonl for every run directory (main: expert, end-to-end at
several latencies, hierarchical; hier_sweep*: hierarchical at long latencies; ball: the
ball-state ablation, conditions tagged "+ball") and results/vla/main/latency.json.
Rates come with Wilson 95% intervals.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

plt.rcParams.update({"font.family": "IPAexGothic", "axes.spines.top": False,
                     "axes.spines.right": False, "axes.grid": True, "grid.alpha": 0.3,
                     "figure.dpi": 150, "savefig.bbox": "tight"})  # fmt: skip
RES = Path("results/vla")
FIG = Path("results/figures")
WORLDS = ("earth", "mars", "moon")
WORLD_JA = {"earth": "地球 (g=9.81)", "mars": "火星 (g=3.71)", "moon": "月 (g=1.62)"}
TIME_SCALE = {"earth": 1.0, "mars": math.sqrt(3.71 / 9.81), "moon": math.sqrt(1.62 / 9.81)}
COLORS = {"earth": "#d1495b", "mars": "#edae49", "moon": "#00798c"}


def wilson(k: int, n: int) -> tuple[float, float]:
    if n == 0:
        return float("nan"), float("nan")
    p, z = k / n, 1.96
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return c - h, c + h


def load() -> pd.DataFrame:
    parts = []
    for f in sorted(RES.glob("*/episodes.jsonl")):
        d = pd.read_json(f, lines=True)
        d["run"] = f.parent.name
        parts.append(d)
    df = pd.concat(parts, ignore_index=True)
    lat = df["condition"].str.extract(r"@([0-9.]+)")[0].astype(float)
    df["latency"] = lat
    df["mode"] = np.where(df["condition"] == "expert", "expert",
                          np.where(df["condition"].str.startswith("hier"), "hier", "vla"))  # fmt: skip
    df["ball_state"] = df["condition"].str.contains(r"\+ball")
    return df


def table(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (world, cond), g in df.groupby(["world", "condition"], sort=False):
        row = {"world": world, "condition": cond, "mode": g["mode"].iloc[0],
               "latency": g["latency"].iloc[0], "ball_state": bool(g["ball_state"].iloc[0]),
               "episodes": len(g)}  # fmt: skip
        for m in ("hit", "in_court", "in_zone"):
            k = int(g[m].sum())
            row[m] = k / len(g)
            row[f"{m}_lo"], row[f"{m}_hi"] = wilson(k, len(g))
        for kind, gk in (("explicit", g[g["kind"] == "explicit"]),
                         ("relational", g[g["kind"] != "explicit"])):  # fmt: skip
            row[f"in_zone_{kind}"] = float(gk["in_zone"].mean()) if len(gk) else float("nan")
            row[f"n_{kind}"] = len(gk)
        md = (
            g.loc[g["hit"] == 0, "min_racket_ball_m"] if "min_racket_ball_m" in g else pd.Series([])
        )
        row["miss_dist_median_m"] = float(md.median()) if len(md) else float("nan")
        row["infer_ms"] = float(g["infer_ms"].mean())
        rows.append(row)
    return pd.DataFrame(rows)


def budget(sub: pd.DataFrame) -> float:
    """Latency at which the hierarchical zone accuracy falls to half of its short-latency value."""
    s = sub.sort_values("latency")
    top = s["in_zone"].iloc[:2].mean()
    half = 0.5 * (top + s["in_zone"].min())
    for (l0, z0), (l1, z1) in zip(s[["latency", "in_zone"]].values, s[["latency", "in_zone"]].values[1:],
                                  strict=False):  # fmt: skip
        if z0 >= half > z1:
            return float(l0 + (z0 - half) * (l1 - l0) / (z0 - z1))
    return float("nan")


def main() -> None:
    df = load()
    lat = json.loads((RES / "main" / "latency.json").read_text())
    real = lat["simulated_real_latency_s"]
    t = table(df)
    budgets = {}
    for w in WORLDS:
        sub = t[(t["world"] == w) & (t["mode"] == "hier") & ~t["ball_state"]]
        if len(sub) >= 3:
            budgets[w] = budget(sub)
    t.to_csv(RES / "summary.csv", index=False)
    out = {"latency": lat, "hier_time_budget_s": budgets, "table": t.to_dict("records")}
    (RES / "summary.json").write_text(
        json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    FIG.mkdir(parents=True, exist_ok=True)

    # 1) controllers per world: hit rate and instructed-zone rate
    def pick(world, cond):
        r = t[(t["world"] == world) & (t["condition"] == cond)]
        return None if r.empty else r.iloc[0]

    rl = f"{real:.2f}"
    groups = [("expert", "物理エキスパート（教師）", "#3d7be0"),
              ("vla@0.00", "VLA 単独・考える時間ゼロ", "#f08c00"),
              (f"vla@{rl}(real)", f"VLA 単独・実測の遅れ {real:.2f} 秒", "#c05800"),
              ("vla@0.00+ball", "VLA 単独＋ボール位置・考える時間ゼロ", "#9c6ade"),
              (f"vla@{rl}(real)+ball", f"VLA 単独＋ボール位置・遅れ {real:.2f} 秒", "#5e3c99"),
              (f"hier@{rl}(real)", f"階層型（VLA が狙い）・遅れ {real:.2f} 秒", "#0b7a3e")]  # fmt: skip
    groups = [g for g in groups if any(pick(w, g[0]) is not None for w in WORLDS)]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.0))
    width = 0.8 / len(groups)
    for ax, m, title in ((axes[0], "hit", "ボールに当てられた割合"),
                         (axes[1], "in_zone", "指示どおりのゾーンに返せた割合")):  # fmt: skip
        for i, (cond, lab, col) in enumerate(groups):
            vals, lo, hi = [], [], []
            for w in WORLDS:
                r = pick(w, cond)
                vals.append(np.nan if r is None else r[m])
                lo.append(0 if r is None else max(r[m] - r[f"{m}_lo"], 0.0))
                hi.append(0 if r is None else max(r[f"{m}_hi"] - r[m], 0.0))
            x = np.arange(len(WORLDS)) + (i - (len(groups) - 1) / 2) * width
            ax.bar(x, vals, width * 0.92, color=col, label=lab, yerr=[lo, hi], capsize=2)
        ax.set_xticks(range(len(WORLDS)), [WORLD_JA[w] for w in WORLDS], fontsize=8)
        ax.set(ylim=(0, 1.32), yticks=np.linspace(0, 1, 6), title=title)
    axes[0].legend(fontsize=7, loc="upper left", ncol=2)
    fig.suptitle(
        "惑星テニス: VLA は打てるか（各条件 100 球、評価用の球出しは学習と別のシード）", fontsize=10
    )
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(FIG / "vla_worlds.png")
    plt.close(fig)

    # 2) hierarchical: zone accuracy vs latency, raw and scaled by the world's time scale
    fig, axes = plt.subplots(1, 2, figsize=(12, 3.8))
    for w in WORLDS:
        sub = t[(t["world"] == w) & (t["mode"] == "hier") & ~t["ball_state"]].sort_values("latency")
        if sub.empty:
            continue
        for ax, scale in ((axes[0], 1.0), (axes[1], TIME_SCALE[w])):
            xs = sub["latency"] * scale
            ax.plot(xs, sub["in_zone"], "o-", color=COLORS[w], label=WORLD_JA[w])
            ax.fill_between(xs, sub["in_zone_lo"], sub["in_zone_hi"], color=COLORS[w], alpha=0.12)
    axes[0].axvline(lat["mean_s"], color="#555", ls="--", lw=1)
    axes[0].text(lat["mean_s"], 0.5, f" A100 の実測\n {lat['mean_s'] * 1000:.0f} ms", fontsize=8)
    axes[0].set(xlabel="VLA の答えが届くまでの遅れ [s]", ylabel="指示どおりのゾーンに返せた割合",
                ylim=(0, 1.05), title="階層型: 答えが遅すぎると狙いが間に合わない")  # fmt: skip
    axes[1].set(xlabel="遅れ × 世界の速さ（地球換算の遅れ）[s]", ylim=(0, 1.05),
                title="世界の速さで割り戻すと 1 本の曲線に重なる")  # fmt: skip
    for ax in axes:
        ax.legend(fontsize=8, loc="upper right")
    fig.tight_layout()
    fig.savefig(FIG / "vla_hier_latency.png")
    plt.close(fig)

    # 3) end-to-end VLA given the exact ball state: re-planning period x latency. At each period
    #    the answer either arrives one period late (latency = period) or at once (latency 0).
    e2e = t[(t["mode"] == "vla") & t["ball_state"]].copy()
    if not e2e.empty:
        per = e2e["condition"].str.extract(r"period([0-9.]+)")[0].astype(float)
        e2e["period"] = np.where(per.notna(), per, np.maximum(e2e["latency"], 0.1))
        fig, axes = plt.subplots(1, 3, figsize=(12, 3.6), sharey=True)
        series = ((False, "o-", "#5e3c99", "答えは 1 周期遅れて届く（遅れ = 周期）"),
                  (True, "s--", "#e66101", "答えはすぐ届く（遅れ 0）"))  # fmt: skip
        for ax, w in zip(axes, WORLDS, strict=True):
            sub = e2e[e2e["world"] == w]
            for fresh, ls, col, lab in series:
                s = sub[(sub["latency"] == 0) == fresh].sort_values("period")
                ax.errorbar(s["period"], s["hit"], yerr=[s["hit"] - s["hit_lo"], s["hit_hi"] - s["hit"]],
                            fmt=ls, color=col, label=lab, capsize=3)  # fmt: skip
            img = t[(t["world"] == w) & (t["mode"] == "vla") & ~t["ball_state"]]
            if not img.empty:
                ax.axhline(img["hit"].max(), color="#888", ls=":", lw=1.2,
                           label="画像のみ（どの遅れでも最大値）")  # fmt: skip
            ax.set(title=WORLD_JA[w], xlabel="新しい行動チャンクに切り替える周期 [s]", ylim=(0, 0.85),
                   xticks=[0.1, 0.2, 0.36])  # fmt: skip
        axes[0].set_ylabel("ボールに当てられた割合")
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="lower center", ncol=3, fontsize=8, frameon=False)
        fig.suptitle("VLA 単独＋ボールの位置・速度を直接入力: 遅れそのものより、行動チャンクの切り替え方で命中率が大きく変わる",
                     fontsize=10)  # fmt: skip
        fig.tight_layout(rect=(0, 0.07, 1, 0.95))
        fig.savefig(FIG / "vla_ball.png")
        plt.close(fig)
    cols = ["world", "condition", "episodes", "hit", "in_court", "in_zone", "in_zone_explicit",
            "in_zone_relational", "miss_dist_median_m", "infer_ms"]  # fmt: skip
    print(t[cols].round(3).to_string(index=False))
    print("hierarchical time budget [s]:", {k: round(v, 2) for k, v in budgets.items()})


if __name__ == "__main__":
    main()
