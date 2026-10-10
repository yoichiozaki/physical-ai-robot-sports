"""Write the results summary between <!-- RESULTS:START --> and <!-- RESULTS:END --> in README.md
(and results/summary.json) from results/eval_main.

    python scripts/update_readme_results.py
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pandas as pd

from tennis.evaluation.experiments import paired_diff

EVAL = Path("results/eval_main")
ORDER = ["A", "B", "B+", "B*", "C", "D-", "D"]
CONDS = {"nominal": "標準", "randomized": "ランダム化", "unknown_wind": "未知: 強風",
         "unknown_court": "未知: コート"}  # fmt: skip


def _key(n: str) -> str:
    return n.split(":")[0]


def main() -> None:
    df = pd.read_csv(EVAL / "matches.csv")
    names = {_key(n): n for n in df["controller"].unique()}
    ctrls = sorted(names.values(), key=lambda n: ORDER.index(_key(n)) if _key(n) in ORDER else 99)
    table = {}
    for c in ctrls:
        g = df[df["controller"] == c]
        table[c] = {
            cond: float(g[g["condition"] == cond]["point_win_rate"].mean()) for cond in CONDS
        }
        table[c]["all"] = float(g["point_win_rate"].mean())
        table[c]["timeouts_per_match"] = float(g["timeouts"].mean()) if "timeouts" in g else None
    pairs = [
        ("D", "B*"),
        ("D", "B+"),
        ("D", "B"),
        ("D", "C"),
        ("D", "D-"),
        ("B*", "B+"),
        ("B", "A"),
    ]
    comps = {}
    for x, y in pairs:
        if x in names and y in names:
            for cond in (None, *CONDS):
                comps[f"{x}-{y}:{cond or 'all'}"] = paired_diff(
                    df, names[x], names[y], "point_win_rate", cond
                )
    Path("results/summary.json").write_text(
        json.dumps({"point_win_rate": table, "paired_differences": comps}, indent=2,
                   ensure_ascii=False), encoding="utf-8")  # fmt: skip

    def fmt(key: str) -> str:
        d = comps[key]
        return f"{d['diff'] * 100:+.1f} pt [95% CI {d['lo'] * 100:+.1f}, {d['hi'] * 100:+.1f}]"

    lines = ["## 結果（要約）", ""]
    if "D-B*:all" in comps:
        lines.append(
            f"- **Hybrid AI（D）** のポイント取得率は全条件で {table[names['D']]['all'] * 100:.1f}%。"
            f"同じ物理・適応層で狙いを手作りルールで決める B* との差 {fmt('D-B*:all')}、"
            f"物理予測制御 B との差 {fmt('D-B:all')}（同じ相手・条件・シードの対応比較）。"
        )
    if "B*-B+:unknown_court" in comps:
        lines.append(
            f"- **オンライン適応**（B* − B+）: 標準 {fmt('B*-B+:nominal')}、未知: 強風 {fmt('B*-B+:unknown_wind')}、"
            f"未知: コート {fmt('B*-B+:unknown_court')}。適応層を外した D− との差 D − D− は "
            f"未知: コートで {fmt('D-D-:unknown_court')}。"
        )
    if "D-C:all" in comps:
        lines.append(
            f"- **純粋な強化学習（C）** は {table[names['C']]['all'] * 100:.1f}%（D − C = {fmt('D-C:all')}）。"
            "ラケット角度から打球の行き先を学ぶのは、同程度の学習量では難しい。"
        )
    if "D-B*:unknown_court" in comps and comps["D-B*:unknown_court"]["hi"] < 0:
        lines.append(
            f"- **限界**: 未知: コートでは学習した戦術が手作りルール + 適応に劣る（D − B* = {fmt('D-B*:unknown_court')}）。"
            "未知の環境での頑健性は主にオンライン同定が担い、戦術そのものの適応は今後の課題。"
        )
    if "B-A:all" in comps:
        lines.append(f"- **物理モデルの効果**（B − A）: {fmt('B-A:all')}。")
    lines += [
        "",
        "ポイント取得率（相手 5 種 × 12 シード × 20 ポイント、評価シードは学習と非重複、未知条件は学習に不使用）:",
        "",
        "| 制御器 | " + " | ".join(CONDS.values()) + " | 全体 |",
        "|---" * (len(CONDS) + 2) + "|",
    ]
    for c in ctrls:
        cells = [f"{table[c][k] * 100:.1f}%" for k in CONDS] + [f"**{table[c]['all'] * 100:.1f}%**"]
        lines.append(f"| {c} | " + " | ".join(cells) + " |")
    lines += [
        "",
        "![point win rate](results/figures/point_win_rate.png)",
        "![drill accuracy](results/figures/drill_accuracy.png)",
    ]
    block = "<!-- RESULTS:START -->\n" + "\n".join(lines) + "\n<!-- RESULTS:END -->"
    readme = Path("README.md")
    text = readme.read_text(encoding="utf-8")
    text = re.sub(
        r"<!-- RESULTS:START -->.*<!-- RESULTS:END -->", lambda _: block, text, flags=re.S
    )
    readme.write_text(text, encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
