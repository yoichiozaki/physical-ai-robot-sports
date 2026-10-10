"""Regenerate all figures (PNG) from the saved raw results (CSV/JSON).

    python scripts/make_figures.py [--eval results/eval_main] [--out results/figures]

Inputs : checkpoints/*/progress.csv, checkpoints/*/run_info.json, <eval>/matches.csv
Outputs: results/figures/*.png, results/compute_cost.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

plt.rcParams.update(
    {
        "font.family": "IPAexGothic",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.3,
        "figure.dpi": 150,
        "savefig.bbox": "tight",
    }
)

COND_JA = {
    "nominal": "標準",
    "randomized": "ランダム化\n(学習分布)",
    "unknown_sticky": "未知: 高摩擦\n低反発",
    "unknown_bouncy": "未知: 低摩擦\n高反発",
}
OPP_JA = {
    "stationary": "静止",
    "random": "ランダム",
    "reactive": "追従",
    "defensive": "守備型",
    "aggressive": "攻撃型",
}
PALETTE = ["#9aa5b1", "#3d7be0", "#1f3f8f", "#8fb8f0", "#f08c00", "#0b7a3e", "#5cc27a"]
ORDER = ["A", "B", "B+", "B*", "C", "D", "D*"]
COLOR = dict(zip(ORDER, PALETTE, strict=True))


def _ctrl_order(df: pd.DataFrame) -> list[str]:
    names = list(df["controller"].unique())
    return sorted(
        names, key=lambda n: ORDER.index(n.split(":")[0]) if n.split(":")[0] in ORDER else 99
    )


def _color(name: str) -> str:
    return COLOR.get(name.split(":")[0], "#555555")


def fig_training_curves(runs: list[Path], out: Path) -> None:
    """Per-run panels (the hybrid counts decisions = shots, the pure RL counts 40 ms steps)."""
    base_path = Path("results/env_baselines.json")
    base = (
        json.loads(base_path.read_text(encoding="utf-8"))["success"] if base_path.exists() else {}
    )
    runs = [r for r in runs if (r / "progress.csv").exists()]
    if not runs:
        return
    fig, axes = plt.subplots(1, len(runs), figsize=(5.6 * len(runs), 3.8), squeeze=False)
    for ax, run in zip(axes[0], runs, strict=True):
        df = pd.read_csv(run / "progress.csv")
        hybrid = "hybrid" in run.name
        unit = "判断回数（= ショット数）" if hybrid else "環境ステップ（40 ms ごと）"
        title = "D: Hybrid AI（戦略選択）" if hybrid else "C: 純粋な強化学習"
        ev = df.dropna(subset=["eval/success_rate"]) if "eval/success_rate" in df else df.iloc[0:0]
        ax.plot(ev["time/total_timesteps"] / 1e6, ev["eval/success_rate"], "o-", ms=3, color="#0b7a3e" if hybrid else "#f08c00",
                label="評価: 1 点勝負の得点率")  # fmt: skip
        if base:
            b = base.get("direct_center_3.5 (= B)")
            best = max(base.values())
            ax.axhline(
                b, ls="--", color="#3d7be0", lw=1, label=f"B 物理予測制御と同じ固定行動 ({b:.2f})"
            )
            ax.axhline(best, ls=":", color="#1f3f8f", lw=1, label=f"最良の固定戦略 ({best:.2f})")
        ax.set(title=title, xlabel=f"{unit} [百万]", ylabel="得点率", ylim=(0, 1))
        if "rollout/ep_rew_mean" in df:
            r = df.dropna(subset=["rollout/ep_rew_mean"])
            ax2 = ax.twinx()
            ax2.plot(
                r["time/total_timesteps"] / 1e6,
                r["rollout/ep_rew_mean"],
                color="#9aa5b1",
                lw=0.8,
                alpha=0.8,
            )
            ax2.set_ylabel("エピソード報酬（灰）", color="#7b8794")
            ax2.grid(False)
        ax.legend(fontsize=7, loc="lower right")
    fig.suptitle(
        "学習曲線（評価環境: Stage 5 = 相手 5 種 + ドメインランダム化、各 60 エピソード）",
        fontsize=10,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.95), w_pad=4.0)
    fig.savefig(out / "training_curves.png")
    plt.close(fig)


def _bar_by_controller(
    df: pd.DataFrame, metric: str, title: str, ylabel: str, out: Path, fname: str
) -> None:
    ctrls = _ctrl_order(df)
    g = df[df["condition"] == "nominal"].groupby("controller")[metric]
    mean = g.mean().reindex(ctrls)
    ci = (1.96 * g.std(ddof=1) / np.sqrt(g.count())).reindex(ctrls)
    fig, ax = plt.subplots(figsize=(9, 3.8))
    bars = ax.bar(
        range(len(ctrls)), mean.values, yerr=ci.values, capsize=4, color=[_color(c) for c in ctrls]
    )
    for b, v in zip(bars, mean.values, strict=True):
        ax.text(
            b.get_x() + b.get_width() / 2,
            v + 0.01,
            f"{v:.2f}",
            ha="center",
            va="bottom",
            fontsize=8,
        )
    ax.set_xticks(range(len(ctrls)), [c.replace(": ", ":\n") for c in ctrls], fontsize=8)
    ax.set(title=title, ylabel=ylabel)
    fig.savefig(out / fname)
    plt.close(fig)


def fig_by_opponent(
    df: pd.DataFrame, metric: str, title: str, ylabel: str, out: Path, fname: str
) -> None:
    ctrls = _ctrl_order(df)
    opps = [o for o in OPP_JA if o in df["opponent"].unique()]
    d = (
        df[df["condition"] == "nominal"]
        .groupby(["controller", "opponent"])[metric]
        .mean()
        .unstack()
    )
    fig, ax = plt.subplots(figsize=(11, 3.8))
    w = 0.8 / len(ctrls)
    for i, c in enumerate(ctrls):
        ax.bar(
            np.arange(len(opps)) + i * w - 0.4 + w / 2,
            d.loc[c, opps].values,
            w,
            label=c,
            color=_color(c),
        )
    ax.set_xticks(range(len(opps)), [OPP_JA[o] for o in opps])
    ax.set(title=title, ylabel=ylabel)
    ax.legend(fontsize=7, ncol=7, loc="upper center", bbox_to_anchor=(0.5, -0.1), frameon=False)
    fig.savefig(out / fname)
    plt.close(fig)


def fig_generalization(df: pd.DataFrame, out: Path) -> None:
    ctrls = _ctrl_order(df)
    conds = [c for c in COND_JA if c in df["condition"].unique()]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for c in ctrls:
        g = df[df["controller"] == c].groupby("condition")
        pts = (g["win"].mean() + 0.5 * g["draw"].mean()).reindex(conds)
        gd = g["goal_diff"].mean().reindex(conds)
        axes[0].plot(range(len(conds)), pts.values, "o-", label=c, color=_color(c))
        axes[1].plot(range(len(conds)), gd.values, "o-", label=c, color=_color(c))
    for ax in axes:
        ax.set_xticks(range(len(conds)), [COND_JA[c] for c in conds], fontsize=8)
    axes[0].set(title="汎化性能: 勝ち点率（勝ち=1, 引分=0.5）", ylabel="勝ち点率", ylim=(0, 1))
    axes[1].set(title="汎化性能: 平均得失点差 / 60 秒", ylabel="得点 − 失点")
    axes[1].legend(fontsize=7, loc="upper right")
    fig.savefig(out / "generalization.png")
    plt.close(fig)


def fig_prediction_error(df: pd.DataFrame, out: Path) -> None:
    sel = df[df["controller"].str.contains("Physics|Hybrid")]
    if sel.empty:
        return
    conds = [c for c in COND_JA if c in sel["condition"].unique()]
    ctrls = _ctrl_order(sel)
    fig, ax = plt.subplots(figsize=(9, 3.6))
    w = 0.8 / len(ctrls)
    for i, c in enumerate(ctrls):
        g = sel[sel["controller"] == c].groupby("condition")["prediction_error_m"]
        m = g.mean().reindex(conds) * 100
        e = (1.96 * g.std(ddof=1) / np.sqrt(g.count())).reindex(conds) * 100
        ax.bar(
            np.arange(len(conds)) + i * w - 0.4 + w / 2,
            m.values,
            w,
            yerr=e.values,
            capsize=3,
            label=c,
            color=_color(c),
        )
    ax.set_xticks(range(len(conds)), [COND_JA[c] for c in conds], fontsize=8)
    ax.set(title="0.5 秒先のパック位置予測誤差（実測との距離）", ylabel="誤差 [cm]")
    ax.legend(fontsize=7, ncol=3)
    fig.savefig(out / "prediction_error.png")
    plt.close(fig)


def fig_strategy_usage(df: pd.DataFrame, out: Path) -> None:
    cols = [c for c in df.columns if c.startswith("strategy_") and c != "strategy_Follow puck"]
    hyb = df[(df["controller"].str.contains("Hybrid")) & (df["condition"] == "nominal")]
    if hyb.empty or not cols:
        return
    hyb = hyb[~hyb["controller"].str.contains("Adapt")]
    labels = {
        "strategy_Defense": "防御",
        "strategy_Direct shot": "ストレート",
        "strategy_Bank shot (L)": "バンク(左)",
        "strategy_Bank shot (R)": "バンク(右)",
    }
    cols = [c for c in labels if c in hyb.columns]
    opps = [o for o in OPP_JA if o in hyb["opponent"].unique()]
    shares = hyb.groupby("opponent")[cols].sum().reindex(opps)
    attack = shares.drop(columns=["strategy_Defense"], errors="ignore")
    attack = attack.div(attack.sum(axis=1), axis=0).fillna(0)
    fig, ax = plt.subplots(figsize=(8, 3.6))
    bottom = np.zeros(len(opps))
    colors = ["#3d7be0", "#f08c00", "#2f9e44"]
    for c, col in zip(attack.columns, colors, strict=False):
        ax.bar(
            [OPP_JA[o] for o in opps], attack[c].values, bottom=bottom, label=labels[c], color=col
        )
        bottom += attack[c].values
    ax.set(
        title="Hybrid AI が選んだ攻撃戦略の内訳（相手別、制御周期ベース）",
        ylabel="割合",
        ylim=(0, 1),
    )
    ax.legend(fontsize=8, loc="upper right")
    fig.savefig(out / "strategy_usage.png")
    plt.close(fig)


def fig_architecture(out: Path) -> None:
    fig, ax = plt.subplots(figsize=(13, 5.4))
    ax.set_xlim(0, 13.4)
    ax.set_ylim(0.2, 5.6)
    ax.axis("off")

    def box(x, y, w, h, text, fc, fs=8.5, bold=False):
        ax.add_patch(
            FancyBboxPatch(
                (x, y),
                w,
                h,
                boxstyle="round,pad=0.03,rounding_size=0.1",
                fc=fc,
                ec="#33415c",
                lw=1.2,
            )
        )
        ax.text(
            x + w / 2,
            y + h / 2,
            text,
            ha="center",
            va="center",
            fontsize=fs,
            weight="bold" if bold else None,
        )

    def arrow(p0, p1, text="", color="#33415c", dy=0.13, rad=0.0):
        ax.add_patch(
            FancyArrowPatch(p0, p1, arrowstyle="-|>", mutation_scale=13, color=color, lw=1.3,
                            connectionstyle=f"arc3,rad={rad}")
        )  # fmt: skip
        if text:
            ax.text(
                (p0[0] + p1[0]) / 2,
                (p0[1] + p1[1]) / 2 + dy,
                text,
                ha="center",
                fontsize=7.5,
                color=color,
            )

    y, h, w, gap = 2.3, 1.3, 1.78, 0.42
    xs = [0.15 + i * (w + gap) for i in range(6)]
    main = [
        ("MuJoCo 物理世界\n1 kHz\nパック・壁・ゴール\n2 軸ガントリー ×2", "#e3ecfa"),
        ("状態推定\n位置・速度センサ\n+ ノイズ\nα-β フィルタ", "#eef2f7"),
        ("物理予測\n減衰 + 壁反射\n(解析解)\nオンライン同定", "#fff1db"),
        ("打撃・防御\nプランナ\n到達時間 vs 移動時間\n打点・助走・法線", "#fff1db"),
        ("モーション制御\n250 Hz\n直線軌道\n速度・加速度・範囲制約", "#f3e8fd"),
        ("速度サーボ\n1 kHz\n力制限付き\nアクチュエータ", "#f3e8fd"),
    ]
    for x, (t, c) in zip(xs, main, strict=True):
        box(x, y, w, h, t, c)
    labels = ["観測", "状態", "軌道", "目標+時刻", "速度"]
    for i, lab in enumerate(labels):
        arrow((xs[i] + w, y + h / 2), (xs[i + 1], y + h / 2))
        ax.text(
            (xs[i] + w + xs[i + 1]) / 2,
            y + h / 2 + 0.22,
            lab,
            ha="center",
            fontsize=7,
            color="#33415c",
        )
    # Hybrid AI above the planner
    hx = xs[3] - 0.55
    box(
        hx,
        4.25,
        w + 1.1,
        1.05,
        "Hybrid AI（PPO, 10 Hz）\n戦略・狙い・速度を選択\n入力: 状態 15 + 物理特徴 12",
        "#d9f2df",
        bold=False,
    )
    arrow((xs[2] + w / 2, y + h), (hx + 0.2, 4.25), "物理特徴", dy=0.05, rad=-0.2)
    arrow(
        (hx + (w + 1.1) / 2, 4.25), (xs[3] + w / 2, y + h), "戦略 (防御/ストレート/バンク)", dy=0.0
    )
    # Pure RL below
    box(
        xs[2] + 0.4,
        0.35,
        w + 1.3,
        1.0,
        "純粋な強化学習（PPO, 25 Hz）\n入力: 状態 15 → 目標位置を直接出力",
        "#d9f2df",
    )
    arrow((xs[1] + w / 2, y), (xs[2] + 0.4, 0.85), "", rad=0.15)
    arrow((xs[2] + 0.4 + w + 1.3, 0.85), (xs[4] + w / 2, y), "目標位置", rad=0.15, dy=-0.3)
    # Feedback: actuator forces act on the physical world
    xr = xs[5] + w + 0.12
    ax.plot(
        [xs[5] + w / 2, xs[5] + w / 2, xr, xr, xs[0] + w / 2],
        [y, 0.7, 0.7, 5.45, 5.45],
        color="#7b8794",
        lw=1.2,
    )
    arrow((xs[0] + w / 2, 5.45), (xs[0] + w / 2, y + h), color="#7b8794")
    ax.text(
        xs[5] + w / 2 - 0.05, 0.45, "関節力 → 物理世界", fontsize=7.5, color="#5c677d", ha="center"
    )
    ax.text(0.2, 4.85, "従来方式（B）: 物理予測 + プランナのみ\nAI 方式（C, D）: 緑の学習モジュールを追加",
            fontsize=8, color="#33415c", va="center")  # fmt: skip
    fig.savefig(out / "architecture.png")
    plt.close(fig)


def compute_cost(runs: list[Path], eval_dir: Path, out_json: Path) -> dict:
    report: dict = {"runs": [], "evaluation": None, "colab_compute_units": 0.0}
    for run in runs:
        info = run / "run_info.json"
        if info.exists():
            d = json.loads(info.read_text(encoding="utf-8"))
            report["runs"].append(
                {
                    "run": d["run_name"],
                    "timesteps": d["total_timesteps"],
                    "duration_min": round(d["training_duration_s"] / 60, 1),
                    "steps_per_s": round(
                        d["total_timesteps"] / max(d["training_duration_s"], 1), 0
                    ),
                    "n_envs": d["n_envs"],
                    "where": "local CPU (Dev Container, 20 threads)",
                }
            )
    meta = eval_dir / "meta.json"
    if meta.exists():
        m = json.loads(meta.read_text(encoding="utf-8"))
        report["evaluation"] = {
            "matches": m["matches"],
            "wall_time_min": round(m["wall_time_s"] / 60, 1),
        }
    out_json.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    return report


def fig_compute(report: dict, out: Path) -> None:
    runs = report["runs"]
    if not runs:
        return
    fig, ax = plt.subplots(figsize=(8, 3.2))
    names = [r["run"] for r in runs]
    ax.barh(names, [r["duration_min"] for r in runs], color="#3d7be0")
    for i, r in enumerate(runs):
        ax.text(
            r["duration_min"],
            i,
            f" {r['timesteps'] / 1e6:.1f}M steps, {r['steps_per_s']:.0f} steps/s",
            va="center",
            fontsize=8,
        )
    ax.set(title="計算コスト（学習時間、ローカル CPU、Colab CU 使用量 0）", xlabel="学習時間 [分]")
    fig.savefig(out / "compute_cost.png")
    plt.close(fig)


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--eval", default="results/eval_main")
    p.add_argument("--out", default="results/figures")
    a = p.parse_args(argv)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    runs = sorted(p for p in Path("checkpoints").glob("ppo_*") if p.is_dir())
    fig_architecture(out)
    fig_training_curves(runs, out)
    report = compute_cost(runs, Path(a.eval), Path("results/compute_cost.json"))
    fig_compute(report, out)
    matches = Path(a.eval) / "matches.csv"
    if matches.exists():
        df = pd.read_csv(matches)
        _bar_by_controller(
            df, "win", "勝率（標準条件、相手 5 種 × 20 試合、95% CI）", "勝率", out, "win_rate.png"
        )
        _bar_by_controller(
            df, "goal_diff", "平均得失点差 / 60 秒（標準条件）", "得点 − 失点", out, "goal_diff.png"
        )
        _bar_by_controller(
            df, "defense_success_rate", "防御成功率（自陣ゴールへの有効シュートを止めた割合）", "成功率", out,
            "defense_success.png",
        )  # fmt: skip
        fig_by_opponent(
            df,
            "goals_for",
            "相手別の平均得点 / 60 秒（標準条件）",
            "得点",
            out,
            "goals_by_opponent.png",
        )
        fig_by_opponent(df, "win", "相手別の勝率（標準条件）", "勝率", out, "win_by_opponent.png")
        fig_generalization(df, out)
        fig_prediction_error(df, out)
        fig_strategy_usage(df, out)
    print("figures:", sorted(f.name for f in out.glob("*.png")))


if __name__ == "__main__":
    main()
