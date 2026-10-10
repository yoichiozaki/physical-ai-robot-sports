"""Regenerate all figures (PNG) from the saved raw results (CSV/JSON).

    python scripts/make_figures.py [--eval results/eval_main] [--out results/figures]

Inputs : checkpoints/ppo_*/progress.csv + run_info.json, <eval>/matches.csv,
         results/residual/drill_eval.json
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
    "unknown_wind": "未知: 強風\n重いボール",
    "unknown_court": "未知: 速い\n滑るコート",
}
OPP_JA = {
    "steady": "安定型",
    "random": "ランダム",
    "aggressive": "攻撃型",
    "lob": "ロブ",
    "slow": "低速",
}
ORDER = ["A", "B", "B+", "B*", "C", "D-", "D"]
PALETTE = ["#9aa5b1", "#3d7be0", "#1f3f8f", "#8fb8f0", "#f08c00", "#5cc27a", "#0b7a3e"]
COLOR = dict(zip(ORDER, PALETTE, strict=True))


def _key(name: str) -> str:
    return name.split(":")[0]


def _ctrl_order(df: pd.DataFrame) -> list[str]:
    names = list(df["controller"].unique())
    return sorted(names, key=lambda n: ORDER.index(_key(n)) if _key(n) in ORDER else 99)


def _color(name: str) -> str:
    return COLOR.get(_key(name), "#555555")


# ----------------------------------------------------------------------------- training
def fig_training_curves(runs: list[Path], out: Path) -> None:
    runs = [
        r for r in runs if (r / "progress.csv").exists() and (r / "progress.csv").stat().st_size > 0
    ]
    if not runs:
        return
    fig, axes = plt.subplots(1, len(runs), figsize=(5.6 * len(runs), 3.6), squeeze=False)
    for ax, run in zip(axes[0], runs, strict=True):
        df = pd.read_csv(run / "progress.csv")
        hybrid = "hybrid" in run.name
        title = (
            "D: Hybrid AI（コースと球速を選択）"
            if hybrid
            else "C: PPO direct（ラケット角度・速度）"
        )
        color = "#0b7a3e" if hybrid else "#f08c00"
        if "rollout/success_rate" in df:
            r = df.dropna(subset=["rollout/success_rate"])
            ax.plot(r["time/total_timesteps"] / 1e3, r["rollout/success_rate"], color=color,
                    alpha=0.35, lw=1, label="学習中（確率的方策）")  # fmt: skip
        if "eval/success_rate" in df:
            ev = df.dropna(subset=["eval/success_rate"])
            ax.plot(ev["time/total_timesteps"] / 1e3, ev["eval/success_rate"], "o-", ms=3,
                    color=color, label="評価（決定的方策, 16 エピソード）")  # fmt: skip
        ax.set(title=title, xlabel="判断回数（= 自分のショット数）[千]",
               ylabel="エピソード勝率（4 ポイント中の過半数）", ylim=(0, 1))  # fmt: skip
        ax.legend(fontsize=7, loc="lower right")
    fig.suptitle(
        "学習曲線（相手 5 種 × 標準/ランダム化条件、1 エピソード = 4 ポイント）", fontsize=10
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94), w_pad=3.0)
    fig.savefig(out / "training_curves.png")
    plt.close(fig)


# ----------------------------------------------------------------------------- matches
def fig_point_win(df: pd.DataFrame, out: Path) -> None:
    ctrls = _ctrl_order(df)
    conds = [c for c in COND_JA if c in set(df["condition"])]
    fig, ax = plt.subplots(figsize=(11, 4.0))
    width = 0.8 / len(ctrls)
    for i, c in enumerate(ctrls):
        means, cis = [], []
        for cond in conds:
            v = df[(df["controller"] == c) & (df["condition"] == cond)]["point_win_rate"].dropna()
            means.append(v.mean())
            cis.append(1.96 * v.std(ddof=1) / np.sqrt(len(v)) if len(v) > 1 else 0)
        x = np.arange(len(conds)) + (i - (len(ctrls) - 1) / 2) * width
        ax.bar(x, means, width * 0.92, yerr=cis, capsize=2, color=_color(c), label=c)
    ax.axhline(0.5, color="#555", lw=0.8, ls="--")
    ax.set_xticks(range(len(conds)), [COND_JA[c] for c in conds])
    ax.set(ylabel="ポイント取得率", ylim=(0, 1.16), yticks=np.linspace(0, 1, 6),
           title="ポイント取得率（相手 5 種 × 12 試合 × 20 ポイント、95% CI）")  # fmt: skip
    ax.legend(fontsize=7.5, ncol=4, loc="upper right")
    fig.savefig(out / "point_win_rate.png")
    plt.close(fig)


def fig_by_opponent(df: pd.DataFrame, out: Path) -> None:
    ctrls = _ctrl_order(df)
    opps = [o for o in OPP_JA if o in set(df["opponent"])]
    nom = df[df["condition"] == "nominal"]
    fig, ax = plt.subplots(figsize=(11, 3.6))
    width = 0.8 / len(ctrls)
    for i, c in enumerate(ctrls):
        g = nom[nom["controller"] == c].groupby("opponent")["point_win_rate"].mean().reindex(opps)
        x = np.arange(len(opps)) + (i - (len(ctrls) - 1) / 2) * width
        ax.bar(x, g.values, width * 0.92, color=_color(c), label=c)
    ax.axhline(0.5, color="#555", lw=0.8, ls="--")
    ax.set_xticks(range(len(opps)), [OPP_JA[o] for o in opps])
    ax.set(
        ylabel="ポイント取得率",
        ylim=(0, 1.16),
        yticks=np.linspace(0, 1, 6),
        title="相手別のポイント取得率（標準条件）",
    )
    ax.legend(fontsize=7.5, ncol=4, loc="upper right")
    fig.savefig(out / "point_win_by_opponent.png")
    plt.close(fig)


def fig_shot_quality(df: pd.DataFrame, out: Path) -> None:
    ctrls = _ctrl_order(df)
    conds = [c for c in COND_JA if c in set(df["condition"])]
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.4))
    for ax, metric, title in (
        (axes[0], "return_rate", "返球成功率（相手の有効打を相手コートへ返した割合）"),
        (axes[1], "in_rate", "自分のショットのイン率"),
    ):
        width = 0.8 / len(ctrls)
        for i, c in enumerate(ctrls):
            m = [
                df[(df["controller"] == c) & (df["condition"] == cond)][metric].mean()
                for cond in conds
            ]
            x = np.arange(len(conds)) + (i - (len(ctrls) - 1) / 2) * width
            ax.bar(x, m, width * 0.92, color=_color(c), label=c)
        ax.set_xticks(range(len(conds)), [COND_JA[c] for c in conds], fontsize=7.5)
        ax.set(ylim=(0, 1), title=title)
        ax.title.set_fontsize(9)
    axes[1].legend(fontsize=6.5, ncol=2, loc="lower left")
    fig.tight_layout()
    fig.savefig(out / "shot_quality.png")
    plt.close(fig)


def fig_identification(df: pd.DataFrame, out: Path) -> None:
    """Online-identified parameters at the end of each match vs. the condition."""
    if "id_e" not in df:
        return
    d = df.dropna(subset=["id_e"])
    d = d[d["controller"].map(_key) == "D"] if (d["controller"].map(_key) == "D").any() else d
    conds = [c for c in COND_JA if c in set(d["condition"])]
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.2))
    items = (
        ("id_e", "コート反発係数 e（推定）", (0.735, "標準モデル"), None),
        ("id_k_drag", "空気抵抗係数 k [1/m]（推定）", (0.02025, "標準モデル"),
         ("unknown_wind", 0.02025 * 1.5, 0.02025 * 1.7)),
        ("wind", "風速 [m/s]（推定）", (0.0, "標準モデル"), ("unknown_wind", 2.0, 2.5)),
    )  # fmt: skip
    for ax, (col, title, (ref, ref_label), truth) in zip(axes, items, strict=True):
        data = []
        for cond in conds:
            g = d[d["condition"] == cond]
            data.append(np.hypot(g["id_wind_x"], g["id_wind_y"]).to_numpy()
                        if col == "wind" else g[col].to_numpy())  # fmt: skip
        ax.boxplot(data, widths=0.5, showfliers=False)
        ax.axhline(ref, color="#3d7be0", ls="--", lw=1, label=ref_label)
        if truth is not None and truth[0] in conds:
            i = conds.index(truth[0]) + 1
            ax.fill_between([i - 0.35, i + 0.35], truth[1], truth[2], color="#f08c00", alpha=0.25,
                            label="真値の範囲（未知: 強風）")  # fmt: skip
        ax.set_xticks(range(1, len(conds) + 1), [COND_JA[c] for c in conds], fontsize=7)
        ax.set_title(title, fontsize=9)
        ax.legend(fontsize=7)
    fig.suptitle("オンライン同定の結果（各試合終了時の推定値、D: Hybrid AI）", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(out / "identification.png")
    plt.close(fig)


# ----------------------------------------------------------------------------- drills
def fig_drill_accuracy(path: Path, out: Path) -> None:
    if not path.exists():
        return
    d = json.loads(path.read_text(encoding="utf-8"))
    conds = [c for c in COND_JA if c in d]
    variants = list(d[conds[0]].keys())
    names = {
        "physics": "物理モデルのみ (B)",
        "physics+residual": "+ 残差学習",
        "physics+online-id": "+ オンライン同定",
        "physics+residual+online-id": "+ 残差学習 + オンライン同定 (B*/D)",
    }
    colors = ["#3d7be0", "#f08c00", "#8fb8f0", "#0b7a3e"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.5))
    width = 0.8 / len(variants)
    for i, v in enumerate(variants):
        x = np.arange(len(conds)) + (i - (len(variants) - 1) / 2) * width
        med = [d[c][v]["landing_error_median_m"] for c in conds]
        inn = [d[c][v]["in_court_rate"] for c in conds]
        axes[0].bar(x, med, width * 0.92, color=colors[i % 4], label=names.get(v, v))
        axes[1].bar(x, inn, width * 0.92, color=colors[i % 4], label=names.get(v, v))
    for ax, t in (
        (axes[0], "狙いと着地点の誤差（中央値）[m]"),
        (axes[1], "打球がコートに入った割合"),
    ):
        ax.set_xticks(range(len(conds)), [COND_JA[c] for c in conds], fontsize=7.5)
        ax.set_title(t, fontsize=9)
    axes[1].set_ylim(0, 1)
    axes[0].legend(fontsize=7)
    fig.suptitle("球出しドリル（640 球/条件、1 セッション = 同じ環境で 40 球）", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(out / "drill_accuracy.png")
    plt.close(fig)


# ----------------------------------------------------------------------------- architecture
def fig_architecture(out: Path) -> None:
    fig, ax = plt.subplots(figsize=(13, 5.6))
    ax.set_xlim(0, 13.4)
    ax.set_ylim(0.0, 5.8)
    ax.axis("off")

    def box(x, y, w, h, text, fc, fs=8.3):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.03,rounding_size=0.1",
                                    fc=fc, ec="#33415c", lw=1.2))  # fmt: skip
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs)

    def arrow(p0, p1, text="", color="#33415c", dy=0.13, rad=0.0):
        ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle="-|>", mutation_scale=13, color=color,
                                     lw=1.3, connectionstyle=f"arc3,rad={rad}"))  # fmt: skip
        if text:
            ax.text((p0[0] + p1[0]) / 2, (p0[1] + p1[1]) / 2 + dy, text, ha="center",
                    fontsize=7.3, color=color)  # fmt: skip

    y, h, w, gap = 2.3, 1.45, 1.78, 0.42
    xs = [0.15 + i * (w + gap) for i in range(6)]
    main = [
        (
            "MuJoCo 物理世界 1 kHz\nボール: 抗力・マグヌス\nバウンド・ネット\n8 自由度ロボット ×2",
            "#e3ecfa",
        ),
        ("観測\n位置・速度センサ\n+ ノイズ\n(50 Hz)", "#eef2f7"),
        ("物理予測\n飛行 RK2 + バウンド\nモデル (同定済み)\n→ 打点候補", "#fff1db"),
        (
            "ショット計画\n着地点 → 打ち出し速度\n→ 打撃モデルの逆解\n→ ラケット向き・速度",
            "#fff1db",
        ),
        ("スイング計画\n閉形式 IK\n5 次多項式軌道\n指定時刻に打点通過", "#f3e8fd"),
        ("位置サーボ\n500 Hz\n慣性で正規化した\nゲイン・トルク制限", "#f3e8fd"),
    ]
    for x, (t, c) in zip(xs, main, strict=True):
        box(x, y, w, h, t, c)
    for i, lab in enumerate(["観測", "状態", "打点・時刻", "姿勢・速度", "関節軌道"]):
        arrow((xs[i] + w, y + h / 2), (xs[i + 1], y + h / 2))
        ax.text((xs[i] + w + xs[i + 1]) / 2, y + h / 2 + 0.22, lab, ha="center", fontsize=6.8,
                color="#33415c")  # fmt: skip
    # learning / adaptation modules
    box(xs[3] - 0.5, 4.35, w + 1.0, 1.15,
        "Hybrid AI（PPO, 1 判断/ショット）\nコース 3 × 深さ 3 × 球速 3 を選択\n入力: ボール・相手・同定値 23 次元",
        "#d9f2df")  # fmt: skip
    arrow((xs[3] + w / 2, 4.35), (xs[3] + w / 2, y + h), "狙う着地点・球速", dy=0.0)
    box(xs[1] - 0.2, 4.35, w + 0.8, 1.15,
        "オンライン同定\nバウンド前後の速度 → e, μ\n飛行中の減速 → 抗力・風\n打球速度 → 打撃ゲイン",
        "#d9f2df")  # fmt: skip
    arrow((xs[1] + w / 2, y + h), (xs[1] + w / 2, 4.35), "", rad=0.0)
    arrow((xs[1] + w + 0.6, 4.9), (xs[2] + w / 2, y + h), "モデル更新", dy=0.05, rad=-0.15)
    box(xs[3] - 0.05, 0.25, w + 1.3, 1.05,
        "残差学習（MLP）\nホームコートでの着地誤差を予測し\n狙いを補正（環境変化の検知時は停止）",
        "#d9f2df")  # fmt: skip
    arrow((xs[3] + 0.6, 1.3), (xs[3] + 0.6, y), "狙い補正", dy=0.0)
    xr = xs[5] + w + 0.12
    ax.plot([xs[5] + w / 2, xs[5] + w / 2, xr, xr, xs[0] + w / 2], [y, 0.12, 0.12, 5.7, 5.7],
            color="#7b8794", lw=1.2)  # fmt: skip
    arrow((xs[0] + w / 2, 5.7), (xs[0] + w / 2, y + h), color="#7b8794")
    ax.text(xs[5] + w / 2 - 0.1, 0.32, "関節トルク → 物理世界", fontsize=7.5, color="#5c677d",
            ha="center")  # fmt: skip
    ax.text(0.25, 1.2, "従来方式（B）: 物理予測 + 計画のみ\nAI 方式（B*, D）: 緑の学習・適応モジュールを追加\n純粋 RL（C）: PPO がラケット角度・速度を直接出力",
            fontsize=8, color="#33415c", va="center")  # fmt: skip
    fig.savefig(out / "architecture.png")
    plt.close(fig)


# ----------------------------------------------------------------------------- compute
def compute_cost(runs: list[Path], eval_dir: Path, out_json: Path) -> dict:
    report: dict = {"runs": [], "evaluation": None, "colab_compute_units": 0.0}
    for run in runs:
        info = run / "run_info.json"
        if info.exists():
            d = json.loads(info.read_text(encoding="utf-8"))
            report["runs"].append(
                {
                    "run": d["run_name"],
                    "decisions": d["total_timesteps"],
                    "duration_min": round(d["training_duration_s"] / 60, 1),
                    "decisions_per_s": round(
                        d["total_timesteps"] / max(d["training_duration_s"], 1), 1
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


# ----------------------------------------------------------------------------- league
def fig_league(d: Path, out: Path, prefix: str = "league", title: str = "") -> None:
    """Elo by generation, tactic evolution and head-to-head of the self-play league."""
    if not (d / "elo.json").exists():
        return
    elo = json.loads((d / "elo.json").read_text(encoding="utf-8"))
    r, ci, steps = elo["ratings"], elo["ci95"], elo["timesteps"]
    gens = sorted((n for n in r if n.startswith("gen_")), key=lambda n: steps[n])
    x = np.array([steps[g] / 1000 for g in gens])
    y = np.array([r[g] for g in gens])
    err = np.array([[r[g] - ci[g][0] for g in gens], [ci[g][1] - r[g] for g in gens]])
    h2h_path = d / "head_to_head.csv"
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(13.5, 4.2), gridspec_kw={"width_ratios": [1.5, 1]})
    if h2h_path.exists():
        hh = pd.read_csv(h2h_path, index_col=0)
        for opp, col, lab in (("gen_000", "#0b7a3e", "対 世代0（出発点）"), ("B: Physics", "#3d7be0", "対 B"),
                              ("B*: Physics+Adapt", "#8fb8f0", "対 B*"), ("opp:aggressive", "#d1495b", "対 攻撃型")):  # fmt: skip
            if opp in hh.columns:
                vals = [hh.loc[g, opp] if g != opp else np.nan for g in gens]
                ax2.plot(x, vals, "o-", color=col, label=lab, lw=1.8 if opp == "gen_000" else 1.2)
        ax2.axhline(0.5, color="#555", ls="--", lw=0.8)
        ax2.set(xlabel="自己対戦での学習量（判断回数, 千）", ylabel="ポイント取得率", ylim=(0, 1.05),
                title="各世代のポイント取得率（相手別）")  # fmt: skip
        ax2.legend(fontsize=7.5, loc="lower right")
    ax.errorbar(x, y, yerr=err, fmt="o-", color="#0b7a3e", capsize=3, lw=2,
                label="自己対戦リーグの各世代（世代0 = " + ("D をサービスボックスで微調整" if prefix == "league_sb" else "D: Hybrid AI") + "）")  # fmt: skip
    refs = (("B*: Physics+Adapt", "#8fb8f0", "B*: 物理+ルール+適応"), ("B+: Physics+Rule", "#1f3f8f", "B+: 物理+ルール"),
            ("B: Physics", "#3d7be0", "B: 物理予測（センター返し）"))  # fmt: skip
    for name, col, lab in refs:
        if name in r:
            ax.axhline(r[name], color=col, ls="--", lw=1.2, label=f"{lab}（{r[name]:.0f}）")
    opp = {n: v for n, v in r.items() if n.startswith("opp:")}
    for n, v in opp.items():
        ax.axhline(v, color="#9aa5b1", lw=0.8, ls=":")
        ax.text(0.01, v, f"相手: {OPP_JA.get(n[4:], n[4:])}", transform=ax.get_yaxis_transform(),
                fontsize=7, color="#7b8794", va="bottom")  # fmt: skip
    ax.set(xlabel="自己対戦での学習量（判断回数, 千）", ylabel="Elo レーティング",
           title=f"自己対戦リーグ{title}: 世代ごとの強さ（総当たり戦のポイント勝敗から推定、95% CI）")  # fmt: skip
    ax.legend(fontsize=7.5, loc="lower right")
    fig.savefig(out / f"{prefix}_elo.png")
    plt.close(fig)
    tac = d / "tactics.csv"
    if tac.exists():
        t = pd.read_csv(tac)
        t = t[t["player"].isin(gens) & t["label"].str.match(r"rl-\d{3}$")]
        if not t.empty:
            fig, axes = plt.subplots(1, 3, figsize=(11, 3.2))
            parts = ((0, "深さ", ("浅い", "中間", "深い")), (1, "左右", ("右", "センター", "左")),
                     (2, "球速", ("遅い", "中速", "速い")))  # fmt: skip
            cols = ("#cfe3f7", "#6aa5e0", "#1f3f8f")
            for ax, (pos, title, names) in zip(axes, parts, strict=True):
                share = np.zeros((len(gens), 3))
                for gi, g in enumerate(gens):
                    tg = t[t["player"] == g]
                    for _, row in tg.iterrows():
                        share[gi, int(row["label"][3 + pos])] += row["count"]
                    share[gi] /= max(share[gi].sum(), 1)
                bottom = np.zeros(len(gens))
                for k in range(3):
                    ax.bar(
                        range(len(gens)), share[:, k], bottom=bottom, color=cols[k], label=names[k]
                    )
                    bottom += share[:, k]
                ax.set_xticks(range(len(gens)), [g.replace("gen_0", "") for g in gens], fontsize=7)
                ax.set(title=f"狙いの{title}", xlabel="世代", ylim=(0, 1))
                ax.legend(fontsize=7, loc="lower right")
            fig.suptitle(
                "自己対戦で戦術はどう変わったか（各世代がホームで打ったショットの内訳）",
                fontsize=10,
            )
            fig.tight_layout(rect=(0, 0, 1, 0.92))
            fig.savefig(out / f"{prefix}_tactics.png")
            plt.close(fig)
    shots = d / "shots.csv"
    if shots.exists():
        s = pd.read_csv(shots)
        s = s[s["receiver"].str.startswith("gen_") | (s["receiver"] == "final")]
        if not s.empty:
            xb = [-3.05, -2.2, -1.4, 0.0]
            yb = [-1.25, -0.4, 0.4, 1.25]
            rate = np.full((3, 3), np.nan)
            cnt = np.zeros((3, 3), dtype=int)
            for i in range(3):
                for j in range(3):
                    g = s[
                        (s["x"] >= xb[i])
                        & (s["x"] < xb[i + 1])
                        & (s["y"] >= yb[j])
                        & (s["y"] < yb[j + 1])
                    ]
                    cnt[i, j] = len(g)
                    if len(g) >= 5:
                        rate[i, j] = g["returned"].mean()
            fig, ax = plt.subplots(figsize=(5.6, 4.6))
            im = ax.imshow(rate, cmap="RdYlGn", vmin=0, vmax=1, origin="upper")
            for i in range(3):
                for j in range(3):
                    txt = (
                        "—" if np.isnan(rate[i, j]) else f"{rate[i, j] * 100:.0f}%\n(n={cnt[i, j]})"
                    )
                    ax.text(j, i, txt, ha="center", va="center", fontsize=8)
            ax.set_xticks(
                range(3), ["フォア側\n(自分の −y)", "センター", "逆側\n(自分の +y)"], fontsize=8
            )
            ax.set_yticks(
                range(3), ["深い\n(ベースライン際)", "中間", "浅い\n(ネット際)"], fontsize=8
            )
            ax.set_title(
                "学習済みロボットが打ち返せた割合\n（相手の打球の着地点別、受け手の座標）",
                fontsize=9,
            )
            ax.grid(False)
            fig.colorbar(im, ax=ax, fraction=0.046)
            fig.savefig(out / f"{prefix}_returns.png")
            plt.close(fig)
    h2h = d / "head_to_head.csv"
    if h2h.exists():
        m = pd.read_csv(h2h, index_col=0)
        keep = [
            n
            for n in [*gens, "B*: Physics+Adapt", "B+: Physics+Rule", "B: Physics"]
            if n in m.index
        ]
        m = m.loc[keep, keep]
        fig, ax = plt.subplots(figsize=(7.5, 6.2))
        im = ax.imshow(m.values, cmap="RdBu", vmin=0.2, vmax=0.8)
        ax.set_xticks(range(len(keep)), [k.split(":")[0] for k in keep], rotation=45, fontsize=7)
        ax.set_yticks(range(len(keep)), [k.split(":")[0] for k in keep], fontsize=7)
        for i in range(len(keep)):
            for j in range(len(keep)):
                v = m.values[i, j]
                if np.isfinite(v):
                    ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=6)
        ax.set_title("総当たり戦のポイント取得率（行が列に対して）", fontsize=9)
        ax.grid(False)
        fig.colorbar(im, ax=ax, fraction=0.04)
        fig.savefig(out / f"{prefix}_h2h.png")
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
    fig_training_curves([r for r in runs if r.name in ("ppo_direct", "ppo_hybrid")], out)
    compute_cost(runs, Path(a.eval), Path("results/compute_cost.json"))
    fig_league(Path("results/league"), out)
    fig_league(Path("results/league_sb"), out, "league_sb", "（サービスボックスあり）")
    fig_drill_accuracy(Path("results/residual/drill_eval.json"), out)
    matches = Path(a.eval) / "matches.csv"
    if matches.exists():
        df = pd.read_csv(matches)
        fig_point_win(df, out)
        fig_by_opponent(df, out)
        fig_shot_quality(df, out)
        fig_identification(df, out)
    print("figures:", sorted(f.name for f in out.glob("*.png")))


if __name__ == "__main__":
    main()
