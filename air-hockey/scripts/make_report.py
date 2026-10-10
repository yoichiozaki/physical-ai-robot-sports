"""Build the Japanese PDF report  reports/説明資料_PAI最終課題_yoichiozaki.pdf  (ReportLab).

Fonts: IPAexGothic / IPAexMincho (TrueType, embedded as subsets). All numbers in tables are
read from the result files (results/eval_main, results/benchmark.json, checkpoints/*/run_info.json,
results/compute_cost.json); the discussion text is in results/conclusion.json.

    python scripts/make_report.py
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.pdfmetrics import registerFontFamily
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    Image,
    KeepTogether,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

OUT = Path("reports/説明資料_PAI最終課題_yoichiozaki.pdf")
FIG = Path("results/figures")
EVAL = Path(__import__("os").environ.get("EVAL_DIR", "results/eval_main"))

pdfmetrics.registerFont(TTFont("Gothic", "/usr/share/fonts/opentype/ipaexfont-gothic/ipaexg.ttf"))
pdfmetrics.registerFont(TTFont("Mincho", "/usr/share/fonts/opentype/ipaexfont-mincho/ipaexm.ttf"))
# <b> in Japanese text is rendered with the Gothic face (no bold TrueType variant is shipped)
registerFontFamily("Mincho", normal="Mincho", bold="Gothic", italic="Mincho", boldItalic="Gothic")
registerFontFamily("Gothic", normal="Gothic", bold="Gothic", italic="Gothic", boldItalic="Gothic")

NAVY = colors.HexColor("#1f3a68")
S = {
    "title": ParagraphStyle(
        "title", fontName="Gothic", fontSize=17, leading=23, textColor=NAVY, alignment=TA_CENTER
    ),
    "sub": ParagraphStyle(
        "sub",
        fontName="Gothic",
        fontSize=9.5,
        leading=14,
        alignment=TA_CENTER,
        textColor=colors.HexColor("#4a5568"),
    ),
    "h1": ParagraphStyle(
        "h1",
        fontName="Gothic",
        fontSize=12.5,
        leading=17,
        textColor=NAVY,
        spaceBefore=7,
        spaceAfter=3,
    ),
    "h2": ParagraphStyle(
        "h2", fontName="Gothic", fontSize=10, leading=14, spaceBefore=4, spaceAfter=1
    ),
    "body": ParagraphStyle(
        "body", fontName="Mincho", fontSize=9, leading=14.2, wordWrap="CJK", spaceAfter=2
    ),
    "bullet": ParagraphStyle(
        "bullet",
        fontName="Mincho",
        fontSize=9,
        leading=13.6,
        wordWrap="CJK",
        leftIndent=10,
        bulletFontName="Gothic",
        bulletIndent=2,
    ),
    "cap": ParagraphStyle(
        "cap",
        fontName="Gothic",
        fontSize=7.8,
        leading=10.5,
        alignment=TA_CENTER,
        textColor=colors.HexColor("#4a5568"),
        wordWrap="CJK",
        spaceAfter=4,
    ),
    "cell": ParagraphStyle("cell", fontName="Gothic", fontSize=7.6, leading=9.8, wordWrap="CJK"),
}
CONTENT_W = A4[0] - 30 * mm


def P(text: str, style: str = "body") -> Paragraph:
    return Paragraph(text, S[style])


def bullets(items: list[str]) -> list[Paragraph]:
    return [Paragraph(t, S["bullet"], bulletText="・") for t in items]


def fig(name: str, caption: str, width: float = 1.0) -> list:
    path = FIG / name
    if not path.exists():
        return [P(f"（図 {name} は未生成）", "cap")]
    img = Image(str(path))
    w = CONTENT_W * width
    img.drawHeight = img.drawHeight * w / img.drawWidth
    img.drawWidth = w
    return [KeepTogether([img, P(caption, "cap")])]


def table(
    rows: list[list], col_w: list[float] | None = None, header: bool = True, caption: str = ""
) -> Table | KeepTogether:
    data = [[Paragraph(str(c), S["cell"]) for c in r] for r in rows]
    t = Table(data, colWidths=col_w, repeatRows=1 if header else 0)
    style = [
        ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#b8c2cc")),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 2),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 2),
    ]
    if header:
        style += [("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#e3ecfa"))]
    t.setStyle(TableStyle(style))
    return KeepTogether([t, P(caption, "cap")]) if caption else t


def _footer(canvas, doc) -> None:
    canvas.saveState()
    canvas.setFont("Gothic", 7.5)
    canvas.setFillColor(colors.HexColor("#718096"))
    canvas.drawString(
        15 * mm,
        9 * mm,
        "Physical AI 応用1 最終課題 / Adaptive Air Hockey AI / OmniCampus: yoichiozaki",
    )
    canvas.drawRightString(A4[0] - 15 * mm, 9 * mm, f"{doc.page}")
    canvas.restoreState()


def results_tables() -> tuple[list[list], list[list], dict]:
    df = pd.read_csv(EVAL / "matches.csv")
    nom = df[df["condition"] == "nominal"]
    rows = [["制御器", "勝率", "得点/60s", "失点/60s", "防御成功率", "予測誤差 0.5s", "判断時間"]]
    stats = {}
    order = ["A", "B", "B+", "B*", "C", "D", "D*"]
    key = lambda n: order.index(n.split(":")[0]) if n.split(":")[0] in order else 99  # noqa: E731
    for name, g in sorted(nom.groupby("controller"), key=lambda kv: key(kv[0])):
        thr = g["threats"].sum()
        dsr = g["saves"].sum() / thr if thr else float("nan")
        pe = g["prediction_error_m"].mean()
        stats[name] = {
            "win": g["win"].mean(),
            "gf": g["goals_for"].mean(),
            "ga": g["goals_against"].mean(),
            "dsr": dsr,
        }
        rows.append(
            [
                name,
                f"{g['win'].mean() * 100:.0f}% (±{1.96 * g['win'].std(ddof=1) / len(g) ** 0.5 * 100:.0f})",
                f"{g['goals_for'].mean():.2f}",
                f"{g['goals_against'].mean():.2f}",
                f"{dsr * 100:.1f}%",
                "—" if pd.isna(pe) else f"{pe * 100:.1f} cm",
                f"{g['latency_ms_mean'].mean():.2f} ms",
            ]
        )
    conds = ["nominal", "randomized", "unknown_sticky", "unknown_bouncy"]
    gen = [["制御器", "標準", "ランダム化", "未知: 高摩擦", "未知: 高反発"]]
    for name, g in sorted(df.groupby("controller"), key=lambda kv: key(kv[0])):
        cells = [name]
        for c in conds:
            gc = g[g["condition"] == c]
            cells.append(
                f"{(gc['win'] + 0.5 * gc['draw']).mean():.2f} / {gc['goal_diff'].mean():+.2f}"
                if len(gc)
                else "—"
            )
        gen.append(cells)
    return rows, gen, stats


def main() -> None:
    bench = json.loads(Path("results/benchmark.json").read_text(encoding="utf-8"))
    cost = json.loads(Path("results/compute_cost.json").read_text(encoding="utf-8"))
    concl = json.loads(Path("results/conclusion.json").read_text(encoding="utf-8"))
    meta = json.loads((EVAL / "meta.json").read_text(encoding="utf-8"))
    res_rows, gen_rows, _ = results_tables()
    s = []
    # ------------------------------------------------------------------ title
    s += [
        P("Adaptive Air Hockey AI", "title"),
        P("物理予測と強化学習を組み合わせ、相手と環境に適応するエアホッケーロボット", "sub"),
        P(
            "Physical AI 応用1 最終課題 ／ OmniCampus アカウント: yoichiozaki ／ 2026 年 10 月",
            "sub",
        ),
        P(concl["links"], "sub"),
        Spacer(1, 5),
    ]
    # ------------------------------------------------------------------ 1
    s += [P("1. 背景と目的", "h1"), P(concl["background"])]
    s += [
        P(
            "研究課題: 従来の物理モデルによる予測制御と比較して、機械学習を用いた戦略選択・予測補正は、対戦性能と未知環境への適応能力を改善できるか？",
            "h2",
        )
    ]
    # ------------------------------------------------------------------ 2
    s += [
        P("2. タスク概要", "h1"),
        P(
            "MuJoCo 上の 2.0 m × 1.0 m のテーブル（ゴール幅 0.30 m）で、自ロボット（青）と相手ロボット（赤）が 60 秒の試合を行う。"
            "ゴール後は失点側からフェイスオフ、パックが片側に 7 秒以上留まった場合は相手側から再開する（ポゼッション規則）。"
            "相手は 5 種（静止・ランダム・追従型・守備型・攻撃型）で、人間らしさのため反応遅延（0.08–0.18 s）と誤った物理モデルを持つ。"
        ),
    ]
    # ------------------------------------------------------------------ 3
    s += [P("3. システム構成", "h1")]
    s += fig(
        "architecture.png",
        "図1 システム構成。判断 50 Hz・軌道生成 250 Hz・速度サーボ 1 kHz の多重レート構成。緑が学習モジュール。",
    )
    s += [
        P(
            "シミュレーション・観測・制御器・学習・評価・可視化・クラウド実行をパッケージ単位で疎結合にし、すべての制御器（従来方式・純粋 RL・Hybrid）"
            "の出力を「マレットの目標位置」に統一した。行動は必ずモーション制御（速度・加速度・可動範囲制約）を経由して関節アクチュエータに渡るため、"
            "学習アルゴリズムを替えても物理シミュレータの変更は不要である。"
        )
    ]
    # ------------------------------------------------------------------ 4
    cal = bench["restitution_calibration"]
    s += [
        P("4. 物理シミュレーション", "h1"),
        P(
            "パックはエアクッションで浮上するため、x・y のスライド関節で平面拘束し、粘性減衰（減衰率 c = 0.15 /s）で空気抵抗と残留摩擦を表した。"
            "衝突形状は同じ高さの球にして接触法線を厳密に水平にし、接触は明示的な接触ペア（法線方向のみ）に限定した。"
            "MuJoCo の dampratio 方式では剛性が 1/(τζ)<super>2</super> に比例し、反発係数 0.85（ζ≈0.05）が 500 Hz で数値的に発散したため、"
            "固有振動数 ω = 600 rad/s を固定して τ = 1/(ωζ) とする方式を考案し、1 kHz で目標反発係数を再現した（表1）。"
        ),
        table(
            [["目標反発係数", "実測 平均", "標準偏差", "範囲", "試行"]]
            + [
                [
                    f"{r['target']:.2f}",
                    f"{r['mean']:.3f}",
                    f"{r['std']:.3f}",
                    f"{r['min']:.3f}–{r['max']:.3f}",
                    r["n"],
                ]
                for r in cal
            ],
            [30 * mm, 28 * mm, 25 * mm, 38 * mm, 18 * mm],
            caption="表1 壁反発係数の校正結果（速度 0.5–6 m/s、入射角 30–90°）。",
        ),
        P(
            "ロボットは実機のエアホッケーロボットで主流の 2 軸直交ガントリー（X: ブリッジ 1.6 kg、Y: キャリッジ 0.4 kg）で、各軸はスライド関節と"
            "力制限付き速度サーボ（1 kHz）で駆動する。速度上限 3 m/s、加速度上限 X 15 / Y 25 m/s<super>2</super>、可動範囲は自陣のみ（中央線を越えない）。"
            "物理テスト 8 種（静止安定・壁反射・マレット衝突・8 m/s での貫通なし・可動範囲・数値安定・ゴール判定・固定シード再現）を自動化した。"
        ),
    ]
    # ------------------------------------------------------------------ 5
    sa = bench["strike_accuracy"]
    pa = bench["prediction_accuracy"]
    s += [
        P("5. ロボット制御（物理予測制御 = 従来方式 B）", "h1"),
        P(
            "パック軌道は区間ごとに解析解 v(t) = v<sub>0</sub>e<super>−ct</super>, p(t) = p<sub>0</sub> + v<sub>0</sub>(1−e<super>−ct</super>)/c で求め、"
            "壁到達時刻を閉形式で計算して法線成分を −e 倍する（予測誤差: 0.25 s 先 {:.1f} cm、0.5 s 先 {:.1f} cm）。"
            "打撃では、パックを方向 d・速さ μ で打ち返すための接触法線 n = (μd − v<sub>in</sub>)/|μd − v<sub>in</sub>| と、法線方向のマレット速度 "
            "s<sub>n</sub> = v<sub>in</sub>·n + |μd − v<sub>in</sub>|/(1+e<sub>m</sub>) を求める。予測軌道上の各候補点について「パック到達時刻」と"
            "「ロボットの停止到達時間 + 助走時間」を比較し、到達不可能な打点は選ばない。バンクショットは反発で傾きが −e 倍になることを用いた閉形式で狙う。".format(
                pa["0.25s"]["mean_m"] * 100, pa["0.5s"]["mean_m"] * 100
            )
        ),
        P(
            "実装上の要点は、(1) 判断周期 50 Hz では打撃開始が最大 20 ms ずれ打点誤差 4 cm（方向誤差 10–30°）になったため、低レベル側で指定時刻に"
            "目標を切り替える「時刻付き打撃指令」を導入、(2) 軸ごとに加速度上限が異なると打撃線が曲がるため方向を保つ協調スルーレート制限、"
            "(3) 打点時刻のコミットメント（ヒステリシス）である。結果、打撃方向誤差は平均 {:.2f}°（90% 点 {:.2f}°）、打点誤差 {:.1f} mm になった。"
            "シュート防御ドリル（2–4.5 m/s）の成功率は追従制御 A がストレート {:.0f}%・バンク {:.0f}%、物理予測制御 B が {:.0f}%・{:.0f}%。".format(
                sa["direction_error_deg_mean"],
                sa["direction_error_deg_p90"],
                sa["intercept_error_m_mean"] * 1000,
                bench["defense_drill_success"]["reactive"]["straight"] * 100,
                bench["defense_drill_success"]["reactive"]["bank"] * 100,
                bench["defense_drill_success"]["physics"]["straight"] * 100,
                bench["defense_drill_success"]["physics"]["bank"] * 100,
            )
        ),
    ]
    # ------------------------------------------------------------------ 6
    s += [P("6. AI の活用方法", "h1")]
    s += bullets(
        [
            "<b>D: Hybrid AI（物理 + 学習）</b> — PPO の方策が「戦略（防御／ストレート／左壁バンク／右壁バンク）× 狙い（ゴール線上 5 点）× 打球速度（3 段階）」"
            "を選び、物理プランナが実行する。入力は状態 15 次元（パック・自分・相手の位置速度、相対位置、ポゼッション時計）と物理特徴 12 次元"
            "（防御線通過予測、最速打点、各シュートの空きレーン幅、バンク可否）。判断はイベント駆動で、1 回の判断が 1 本のシュートに対応する。",
            "<b>C: 純粋な強化学習</b> — PPO がマレットの目標位置（2 次元連続値）を 25 Hz で直接出力する。物理予測を使わない比較対象。",
            "<b>適応（B*, D*）</b> — 自由飛行区間の ln|v| 回帰で減衰率、側壁バウンドで反発係数をオンライン同定し、事前値と証拠重みで混合して予測モデルを更新する。"
            "センサノイズには物理モデル予測付きの α-β フィルタ（衝突でリセット）を用いる。",
        ]
    )
    # ------------------------------------------------------------------ 7
    runs = {r["run"]: r for r in cost["runs"]}
    run_rows = [["学習", "総ステップ", "学習時間", "速度", "並列環境"]]
    for name, r in runs.items():
        run_rows.append(
            [
                name,
                f"{r['timesteps']:,}",
                f"{r['duration_min']:.0f} 分",
                f"{r['steps_per_s']:.0f} /s",
                r["n_envs"],
            ]
        )
    s += [
        P("7. 学習手法", "h1"),
        P(
            "Stable-Baselines3 の PPO（MlpPolicy）。純粋 RL は 256×256、Hybrid は 128×128 の方策・価値ネットワーク。カリキュラムは "
            "Stage 1 静止パック打撃 → 2 直線シュート防御 → 3 壁反射シュート防御 → 4 単純な相手との対戦 → 5 複数相手 + ドメインランダム化 → "
            "6 未知条件評価（学習に不使用）。ドメインランダム化は減衰率 0.05–0.35、壁反発 0.75–0.92、マレット反発 0.6–0.8、相手速度 0.8–1.1 倍、"
            "観測ノイズ（位置 3 mm・速度 0.03 m/s）。純粋 RL の報酬はゴール ±10、打撃 +0.3、枠内打撃 +0.7、防御 +1、自陣方向への打撃 −0.5、"
            "エネルギー・範囲外ペナルティと、最適方策を変えないポテンシャル型整形。Hybrid はゴール ±10、枠内 +0.5、防御 +0.5、自陣方向 −0.5 のみ。"
            "学習はすべてローカル CPU（Dev Container）で行い、Colab の計算ユニット消費は 0 である（理由は 14 章）。"
        ),
        table(
            run_rows,
            [42 * mm, 30 * mm, 25 * mm, 25 * mm, 22 * mm],
            caption="表2 学習の計算コスト（results/compute_cost.json）。Hybrid のステップ = 判断（ショット）回数。",
        ),
    ]
    s += fig(
        "training_curves.png",
        "図2 学習曲線（Stage 5 環境での 1 点勝負の得点率と報酬）。破線は B と同じ固定行動、点線は最良の固定戦略（results/env_baselines.json）。",
        0.95,
    )
    # ------------------------------------------------------------------ 8
    s += [
        P("8. 実験設定", "h1"),
        P(
            "制御器 A 追従制御、B 物理予測制御、B+ 物理 + ルール（相手から最も遠いコースを選ぶ幾何ヒューリスティック）、B* 物理 + 適応、"
            "C 純粋 RL、D Hybrid AI、D* Hybrid AI + 適応 を、相手 5 種 × 条件 4 種（標準、学習分布のランダム化、未知: 高摩擦・低反発、未知: 低摩擦・高反発。"
            "未知条件は相手速度 1.15–1.3 倍・観測ノイズ 2 倍以上も含む）× 20 試合（各 60 秒）で評価した（計 {} 試合、評価用シード 10000 番台は学習と重複しない）。"
            "指標は勝率、得点・失点、防御成功率（自陣ゴールに入る軌道のシュートを止めた割合）、ラリー長、予測誤差、判断時間。".format(
                meta["matches"]
            )
        ),
    ]
    # ------------------------------------------------------------------ 9
    s += [
        P("9. 実験結果", "h1"),
        table(
            res_rows,
            [40 * mm, 22 * mm, 19 * mm, 19 * mm, 22 * mm, 24 * mm, 20 * mm],
            caption="表3 標準条件の結果（相手 5 種 × 20 試合の平均。勝率の括弧は 95% 信頼区間の半幅）。",
        ),
        P(concl["results_text"]),
    ]
    s += fig("win_by_opponent.png", "図3 相手別の勝率（標準条件）。", 0.95)
    s += fig("goals_by_opponent.png", "図4 相手別の平均得点（標準条件）。", 0.95)
    s += [
        table(
            gen_rows,
            [40 * mm, 33 * mm, 33 * mm, 33 * mm, 33 * mm],
            caption="表4 条件別の 勝ち点率 / 平均得失点差（相手 5 種 × 20 試合）。",
        )
    ]
    s += fig("generalization.png", "図5 未知条件への汎化（勝ち点率と得失点差）。", 0.95)
    s += fig(
        "prediction_error.png",
        "図6 0.5 秒先の予測誤差（物理予測を使う制御器）。適応付き（*）で未知条件の誤差が約半分。",
        0.8,
    )
    s += fig("strategy_usage.png", "図7 Hybrid AI が相手ごとに選んだ攻撃戦略の割合。", 0.75)
    # ------------------------------------------------------------------ 10
    s += [P("10. 技術的工夫", "h1")]
    s += bullets(concl["techniques"])
    # ------------------------------------------------------------------ 11
    s += [P("11. 考察", "h1"), P(concl["discussion"])]
    s += [P("12. 制限事項", "h1")]
    s += bullets(concl["limitations"])
    s += [P("13. 今後の展望", "h1")]
    s += bullets(concl["future"])
    s += [
        P("14. 再現性", "h1"),
        P(
            "すべて VS Code Dev Container（Python 3.11.17, MuJoCo 3.15.0, Gymnasium 1.4.0, Stable-Baselines3 2.9.0, PyTorch 2.14.1 CPU）で実行でき、"
            "依存関係は requirements/lock.txt に固定した。学習: <font name='Gothic'>python scripts/train.py --config configs/training/ppo_hybrid.yaml</font>、"
            "評価: <font name='Gothic'>python scripts/evaluate.py --config configs/evaluation/main.yaml</font>、図・動画・本資料: "
            "<font name='Gothic'>make figures video report</font>。テスト（pytest 30 件以上）と Ruff を CI で実行する。"
            "Colab は公式 google-colab-cli 0.7.4 の手順（承認ゲート付き）を用意したが、MLP の PPO は環境計算が律速で CPU 並列の方が速いため本研究では使用していない。"
        ),
    ]
    OUT.parent.mkdir(parents=True, exist_ok=True)
    doc = SimpleDocTemplate(
        str(OUT),
        pagesize=A4,
        leftMargin=15 * mm,
        rightMargin=15 * mm,
        topMargin=13 * mm,
        bottomMargin=15 * mm,
        title="Adaptive Air Hockey AI - 説明資料",
        author="yoichiozaki",
        subject="Physical AI 応用1 最終課題",
    )
    doc.build(s, onFirstPage=_footer, onLaterPages=_footer)
    print(OUT)


if __name__ == "__main__":
    main()
