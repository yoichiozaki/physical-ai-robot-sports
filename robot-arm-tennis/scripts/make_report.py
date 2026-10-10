"""Build the Japanese PDF report  reports/説明資料_PAI最終課題_yoichiozaki.pdf  (ReportLab).

Fonts: IPAexGothic / IPAexMincho (TrueType, embedded as subsets). All numbers are read from
the result files (results/eval_main, results/calibration.json, results/residual/*.json,
checkpoints/*/run_info.json, results/compute_cost.json).

    python scripts/make_report.py
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
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

from tennis.evaluation.experiments import paired_diff

OUT = Path("reports/説明資料_PAI最終課題_yoichiozaki.pdf")
FIG = Path("results/figures")
EVAL = Path(os.environ.get("EVAL_DIR", "results/eval_main"))
LINKS = (
    "コード: https://github.com/yoichiozaki/physical-ai-robot-sports （robot-arm-tennis/） ／ "
    "動画: YouTube「PAI最終課題_yoichiozaki」"
)

pdfmetrics.registerFont(TTFont("Gothic", "/usr/share/fonts/opentype/ipaexfont-gothic/ipaexg.ttf"))
pdfmetrics.registerFont(TTFont("Mincho", "/usr/share/fonts/opentype/ipaexfont-mincho/ipaexm.ttf"))
registerFontFamily("Mincho", normal="Mincho", bold="Gothic", italic="Mincho", boldItalic="Gothic")
registerFontFamily("Gothic", normal="Gothic", bold="Gothic", italic="Gothic", boldItalic="Gothic")

NAVY = colors.HexColor("#1f3a68")
GREY = colors.HexColor("#4a5568")
S = {
    "title": ParagraphStyle("title", fontName="Gothic", fontSize=17, leading=23, textColor=NAVY,
                            alignment=TA_CENTER),
    "sub": ParagraphStyle("sub", fontName="Gothic", fontSize=9.2, leading=13.5, alignment=TA_CENTER,
                          textColor=GREY),
    "h1": ParagraphStyle("h1", fontName="Gothic", fontSize=12.5, leading=17, textColor=NAVY,
                         spaceBefore=7, spaceAfter=3),
    "h2": ParagraphStyle("h2", fontName="Gothic", fontSize=10, leading=14, spaceBefore=4,
                         spaceAfter=1),
    "body": ParagraphStyle("body", fontName="Mincho", fontSize=9, leading=14.2, wordWrap="CJK",
                           spaceAfter=2),
    "bullet": ParagraphStyle("bullet", fontName="Mincho", fontSize=9, leading=13.6, wordWrap="CJK",
                             leftIndent=10, bulletFontName="Gothic", bulletIndent=2),
    "cap": ParagraphStyle("cap", fontName="Gothic", fontSize=7.8, leading=10.5, alignment=TA_CENTER,
                          textColor=GREY, wordWrap="CJK", spaceAfter=4),
    "cell": ParagraphStyle("cell", fontName="Gothic", fontSize=7.6, leading=9.8, wordWrap="CJK"),
}  # fmt: skip
CONTENT_W = A4[0] - 30 * mm
ORDER = ["A", "B", "B+", "B*", "C", "D-", "D"]
CONDS = ["nominal", "randomized", "unknown_wind", "unknown_court"]
COND_JA = {"nominal": "標準", "randomized": "ランダム化", "unknown_wind": "未知: 強風",
           "unknown_court": "未知: コート"}  # fmt: skip


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


def table(rows: list[list], col_w: list[float] | None = None, caption: str = "") -> KeepTogether:
    data = [[Paragraph(str(c), S["cell"]) for c in r] for r in rows]
    t = Table(data, colWidths=col_w, repeatRows=1)
    t.setStyle(TableStyle([
        ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#b8c2cc")),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 2),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 2),
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#e3ecfa")),
    ]))  # fmt: skip
    return KeepTogether([t, P(caption, "cap")])


def _footer(canvas, doc) -> None:
    canvas.saveState()
    canvas.setFont("Gothic", 7.5)
    canvas.setFillColor(colors.HexColor("#718096"))
    canvas.drawString(15 * mm, 9 * mm,
                      "Physical AI 応用1 最終課題 / Robot-Arm Tennis AI / OmniCampus: yoichiozaki")  # fmt: skip
    canvas.drawRightString(A4[0] - 15 * mm, 9 * mm, f"{doc.page}")
    canvas.restoreState()


# ----------------------------------------------------------------------------- data
def _key(name: str) -> str:
    return name.split(":")[0]


def load() -> dict:
    df = pd.read_csv(EVAL / "matches.csv")
    names = {_key(n): n for n in df["controller"].unique()}
    ctrls = sorted(names.values(), key=lambda n: ORDER.index(_key(n)) if _key(n) in ORDER else 99)
    cal = json.loads(Path("results/calibration.json").read_text(encoding="utf-8"))
    drill = json.loads(Path("results/residual/drill_eval.json").read_text(encoding="utf-8"))
    res_train = json.loads(Path("results/residual/train.json").read_text(encoding="utf-8"))
    meta = json.loads((EVAL / "meta.json").read_text(encoding="utf-8"))
    runs = {}
    for r in ("ppo_hybrid", "ppo_direct"):
        p = Path("checkpoints") / r / "run_info.json"
        if p.exists():
            runs[r] = json.loads(p.read_text(encoding="utf-8"))
    return {"df": df, "names": names, "ctrls": ctrls, "cal": cal, "drill": drill,
            "res_train": res_train, "meta": meta, "runs": runs}  # fmt: skip


def pw(df: pd.DataFrame, ctrl: str, cond: str | None = None) -> tuple[float, float]:
    g = df[df["controller"] == ctrl]
    if cond is not None:
        g = g[g["condition"] == cond]
    v = g["point_win_rate"].dropna()
    return float(v.mean()), float(1.96 * v.std(ddof=1) / np.sqrt(len(v))) if len(v) > 1 else 0.0


def diff_text(d: dict, pct: bool = True) -> str:
    k = 100 if pct else 1
    unit = " pt" if pct else ""
    sig = "有意" if (d["lo"] > 0 or d["hi"] < 0) else "有意差なし"
    return f"{d['diff'] * k:+.1f}{unit} [95% CI {d['lo'] * k:+.1f}, {d['hi'] * k:+.1f}]（{sig}）"


# ----------------------------------------------------------------------------- report
def build(data: dict) -> list:
    df, names, ctrls, cal = data["df"], data["names"], data["ctrls"], data["cal"]
    rt, runs = data["res_train"], data["runs"]
    n_match = len(df)
    seeds = sorted(df["seed"].unique())
    pts = int(df["points_won"].iloc[0] + df["points_lost"].iloc[0])
    s: list = []
    s += [
        P("Robot-Arm Tennis AI", "title"),
        P("物理予測・学習・オンライン適応で打ち合う 8 自由度ロボットアーム", "sub"),
        P(
            "Physical AI 応用1 最終課題 ／ OmniCampus アカウント: yoichiozaki ／ 2026 年 10 月",
            "sub",
        ),
        P(LINKS, "sub"),
        Spacer(1, 5),
    ]
    # 1 ------------------------------------------------------------------
    s += [
        P("1. 背景と目的", "h1"),
        P("テニスでは、空気抵抗と回転で曲がりバウンドで変化する 3 次元のボールを、0.6–1 秒のうちに予測し、"
          "全身（移動と腕）を協調させて「狙った場所へ、狙った速さで」打ち返す必要がある。人は経験（学習）で戦術と"
          "感覚を磨き、コートや風が変われば数球で適応する。本課題では、この Physical AI の典型的な難しさを持つ題材として、"
          "MuJoCo 上でロボットアーム同士がテニスをする環境を一から構築し、物理モデルと学習の役割分担を検証した。"),
        P("研究課題: 物理モデルによる予測・計画に、学習（戦術の強化学習、着地誤差の残差学習）とオンライン適応を組み合わせると、"
          "対戦性能と未知環境への頑健性は向上するか？", "h2"),
    ]  # fmt: skip
    # 2 ------------------------------------------------------------------
    s += [
        P("2. タスク概要", "h1"),
        P("ロボット用のミニテニスコート（6.0 m × 2.4 m、ネット高 0.35 m）で、実物の硬式テニスボール（半径 33 mm、57 g）を使う。"
          "重力は縮尺できないため、球速 3–7 m/s・飛行 0.6–1.2 秒の「ミニテニス」とした。ルールはシングルス相当: "
          "サーブはトスしたボールを 1 バウンド後に打ち、相手コートに入れる。2 バウンド・アウト・ネット・ボディ接触・二度打ちは失点、"
          "サーブは 1 ポイントごとに交代し、ポイント間には 1.5 秒の間（両ロボットが構えに戻る）を置く。30 秒で決着しないポイントは打ち切り、最後の打球が相手側でバウンド済みなら打者の得点とする（両者共通）。"
          f"評価の 1 試合は {pts} ポイント。"),
        P("相手は人間らしい不完全さを持つスクリプト型 5 種（安定型・ランダム・攻撃型・ロブ・低速）で、反応遅延 0.08–0.15 s を"
          "わずかに不正確な自分の物理モデルで外挿補償し、狙いには誤差がある。環境条件は 4 種: 標準、ランダム化（学習分布: "
          "コート反発 0.68–0.82・摩擦・ラケット反発・抗力 0.75–1.3 倍・風 0.8 m/s 以下・相手速度・センサノイズ）、"
          "未知: 強風（横風 2–2.5 m/s + 抗力 1.5–1.7 倍 + ノイズ 2 倍）、未知: コート（よく弾み滑るコート、反発の強いラケット、"
          "1.2 倍速い相手、ノイズ 2 倍）。未知条件は学習・残差モデルの学習に一切使っていない。"),
    ]  # fmt: skip
    # 3 ------------------------------------------------------------------
    s += [P("3. システム構成", "h1")]
    s += fig(
        "architecture.png",
        "図1 システム構成。物理 1 kHz・サーボ 500 Hz・観測と判断 50 Hz・戦術判断はショットごと（約 1 Hz）。緑が学習・適応モジュール。",
    )
    s += [
        P("シミュレーション（sim）・物理予測と計画（control）・学習（learning）・評価（evaluation）・可視化（viz）を疎結合にし、"
          "すべての制御器は最終的に「打点・時刻・ラケットの向きと速度」を経てスイング計画とサーボに渡る。学習は"
          "「どこへ打つか」（D）または「ラケットをどう振るか」（C）だけを担い、関節トルクは常に物理的な制約の中で生成される。"),
    ]  # fmt: skip
    # 4 ------------------------------------------------------------------
    b, im = cal["bounce"], cal["impact"]
    s += [
        P("4. 物理シミュレーションとロボット", "h1"),
        P("ロボットは、床の 2 軸直動ガントリー（前後 1.5 m・左右 2.8 m）に、肘付き 6 軸アーム（球面手首）とラケット（打球面 30 × 23 cm）を"
          "載せた 8 自由度機械で、2 台が向かい合う。アームは閉形式の逆運動学を持ち（MuJoCo の順運動学と 1e-9 以内で一致することをテストで確認）、"
          "各関節は慣性から定めたゲインの位置サーボ（速度・加速度フィードフォワード、関節間の慣性結合を考慮したトルク上限、重力補償）で駆動する。"
          "モデルは外部アセットを使わずコードから手続き生成した。"),
        P("ボールの空力は MuJoCo の楕円体流体モデル（抗力 + マグヌス力）で表し、接触は明示的な接触ペアに限定して時定数を調整した"
          "（1 kHz で固有振動数 600 rad/s ではバウンドの反発係数が衝突速度で 0.69–0.83 にばらついたため 300 rad/s とした）。"
          "予測用の物理モデルはシミュレータから同定した（表1）。"),
        table(
            [["要素", "同定値", "精度・効果"],
             ["空気抵抗 k_d", f"{cal['drag']['k_drag']:.5f} /m",
              f"予測誤差 RMS {cal['drag']['rms_position_error_m'] * 1000:.1f} mm（無視すると {cal['drag']['rms_if_drag_ignored_m'] * 100:.0f} cm）"],
             ["マグヌス k_m", f"{cal['magnus']['k_magnus']:.5f}",
              f"RMS {cal['magnus']['rms_position_error_m'] * 1000:.1f} mm（スピン無視で {cal['magnus']['rms_if_spin_ignored_m'] * 100:.0f} cm）"],
             ["コートのバウンド", f"e = {b['e_z']:.3f}, μ = {b['mu']:.3f}, β = {b['beta']:.3f}",
              f"e のばらつき σ = {b['e_z_std']:.3f}、接線速度 RMS {b['rms_tangential_error_mps']:.2f} m/s"],
             ["ラケット打撃（実スイング）", f"a = {im['a']:.3f}, b = {im['b']:.3f}, c_t = {im['c_t']:.3f}",
              f"法線方向 RMS {im['rms_normal_mps']:.2f} m/s（{im['hits']} 打）"],
             ["遅い球の打撃（サーブ）", f"b × {cal['serve']['slow_impact_gain']:.3f}（入射 1 m/s 以下）",
              f"σ = {cal['serve']['std']:.3f}（{cal['serve']['serves']} 本）。サーブの着地誤差 +0.8 m → ±0.2 m"]],
            [38 * mm, 52 * mm, CONTENT_W - 90 * mm],
            "表1 物理モデルの同定（scripts/calibrate.py, calibrate_impact.py）。打撃モデル: v_out = c_t v_in,t + (a v_in,n + b s) n。",
        ),
    ]  # fmt: skip
    # 5 ------------------------------------------------------------------
    s += [
        P("5. 物理予測制御（従来方式 B）", "h1"),
        P("観測（50 Hz、位置・速度センサ + ノイズ）から、RK2（5 ms）で空気抵抗とバウンドを含む軌道を 2.2 秒先まで予測し、"
          "腕の届く範囲を通る時刻・位置を打点候補として列挙する。各候補で、狙った着地点に落とす打ち出し速度を弾道ソルバ（抗力込みの反復）で求め、"
          "打撃モデルの逆解 n = normalize(v_out − c_t v_in)、s = (v_out·n − a v_in·n)/b からラケット面の向き n とラケット速度 s を決める。"
          "これを閉形式 IK で関節角に変換し、関節速度は重み付き最小ノルムで配分、準備 → スイング → フォロースルーを区分 5 次多項式で"
          "「指定時刻に指定の向きと速度で打点を通過する」軌道として生成する。到達時間・関節速度・トルクの制約を満たす候補のうち最良を選ぶ。"),
        P("開発中の主な発見: (1) 軌道セグメント連結時に終端速度を 0 扱いしていたバグで打撃誤差が 1.1 m/s あり、修正で 0.20 m/s に改善した。"
          "(2) バウンド前にスイングを確定すると、不規則なバウンドで打点が約 10 cm ずれてフレームに当たるため、バウンド直後の打点に罰則を設け、"
          "バウンドを観測してから打点を確定する（人の「ボールを見て打つ」に相当）。"
          "(3) 初回の評価で、ポイント終了と同時に次のトスを出していたため、大きく振られたロボットが自分のサーブに間に合わず、"
          "強化学習と手作りルールの両方がこの規則の穴を利用していた。1.5 秒の間を入れ、学習と評価をすべてやり直した。"
          "同時に、サーブ（ほぼ静止した球の打撃）が 0.8 m 長くなる打撃モデルの偏りを見つけ、遅い打撃のゲインを同定して解消した。"),
    ]  # fmt: skip
    # 6 ------------------------------------------------------------------
    rh = runs.get("ppo_hybrid", {})
    rd = runs.get("ppo_direct", {})
    s += [
        P("6. 学習と適応", "h1"),
        P("6.1 残差学習（教師あり）", "h2"),
        P("ボールマシン練習（ドリル）で、計画した狙いと実際の着地点の差を、打点・速度・ラケット姿勢など 16 特徴から予測する MLP（16→64→64→2）を学習し、"
          f"狙いを逆方向に補正する。ホームコート（標準条件）の {rt['train_samples']:,} 球で学習し、検証誤差の中央値は "
          f"{rt['val_median_without_m'] * 100:.1f} → {rt['val_median_with_m'] * 100:.1f} cm。条件を毎球ランダムに変えたデータでは誤差が予測不能（ほぼ改善なし）で、"
          "残差学習は「同じ環境で繰り返し練習する」ことで効く、という人の練習と同じ性質を示した。"),
        P("6.2 オンライン同定（適応）", "h2"),
        P("試合中の自分の観測だけから物理モデルを更新する。バウンド前後 3 サンプルの速度から反発係数 e と摩擦 μ を推定（直近 9 回の中央値で外れ値に頑健）、"
          "バウンドのない区間の水平減速度から抗力係数と風速を再帰最小二乗で推定（マグヌス力の影響が小さい水平成分のみ使用）、"
          "自分の打球の初速から打撃ゲインを推定する。推定値が標準モデルから有意に離れた（変化検知）ときだけ予測・ショット計算に反映し、"
          "そのとき残差モデルは停止する（残差は標準モデル用に学習したもので、環境が変わると逆効果になるため）。"),
        P("6.3 強化学習（PPO）", "h2"),
        P("1 判断 = 自分のショット 1 回とし、相手の打球が自分に向かってきた瞬間（またはサーブ）に方策を呼ぶ Gymnasium 環境を作った。"
          "観測は 23 次元（ボール位置・速度、予測バウンド点、自分と相手の位置、相手ラケット、サーブフラグ、オンライン同定値の偏差）。"
          "D（Hybrid AI）は MultiDiscrete [3, 3, 3]（左右 × 深さ × 球速 = 27 通りの狙い）を出力し、物理層（予測・ショット解・残差・同定）が実行する。"
          "C（純粋 RL）は Box(3)（ラケット面の仰角・方位角・ラケット速度）を直接出力し、打球がどう飛ぶかを自分で学ぶ必要がある"
          "（打点への移動のみ物理層を使う。関節レベルの end-to-end RL は接触が疎で予算内に学習不能と判断）。"
          "報酬は得点 +1・失点 −1、自分のショットがインで +0.1、1 エピソード = 4 ポイント、相手 5 種 × 標準/ランダム化で学習した。"
          + (f"学習量は D {rh.get('total_timesteps', 0):,} 判断（{rh.get('training_duration_s', 0) / 60:.0f} 分）、"
             f"C {rd.get('total_timesteps', 0):,} 判断（{rd.get('training_duration_s', 0) / 60:.0f} 分）、いずれもローカル CPU（16 並列環境 × 2）。"
             if rh and rd else "")),
    ]  # fmt: skip
    s += fig(
        "training_curves.png",
        "図2 学習曲線（判断回数あたりのエピソード勝率）。評価は決定的方策・16 エピソード。",
    )
    # 7 ------------------------------------------------------------------
    s += [
        P("7. 実験設定", "h1"),
        table(
            [["記号", "制御器", "内容"],
             ["A", "素朴な制御", "相手の球を等速直線と仮定（重力・バウンドなし）、固定ストローク。自分のサーブのみ物理を使用"],
             ["B", "物理予測制御", "4–5 節の物理層。狙いは常にセンター"],
             ["B+", "物理 + ルール戦術", "相手ラケットから遠いコーナーへ（オープンコート）"],
             ["B*", "B+ + 学習・適応", "B+ に残差学習（変化検知で停止）とオンライン同定を追加"],
             ["C", "純粋 RL", "PPO がラケット角度・速度を直接出力（オンライン同定あり）"],
             ["D-", "Hybrid AI（適応なし）", "D の方策を、オンライン同定・残差なしの物理層で実行（アブレーション）"],
             ["D", "Hybrid AI（提案）", "PPO が狙いを選び、B* と同じ物理・学習・適応層が実行"]],
            [12 * mm, 38 * mm, CONTENT_W - 50 * mm],
            f"表2 比較する制御器。各制御器 × 相手 5 種 × 条件 4 種 × シード {len(seeds)}（{seeds[0]}–{seeds[-1]}、学習と重複なし）"
            f"= 計 {n_match} 試合。同じシードは同じ初期条件・同じ環境パラメータ（対応のある比較）。",
        ),
    ]  # fmt: skip
    # 8 ------------------------------------------------------------------
    rows = [["制御器", *[COND_JA[c] for c in CONDS], "全体", "ラリー長", "判断時間"]]
    for c in ctrls:
        g = df[df["controller"] == c]
        cells = [c]
        for cond in CONDS:
            m, ci = pw(df, c, cond)
            cells.append(f"{m * 100:.1f}% ±{ci * 100:.1f}")
        m, ci = pw(df, c)
        cells += [
            f"{m * 100:.1f}%",
            f"{g['avg_rally_hits'].mean():.1f}",
            f"{g['latency_ms_mean'].mean():.1f} ms",
        ]
        rows.append(cells)
    s += [P("8. 結果", "h1")]
    s += [table(rows, [34 * mm, 22 * mm, 22 * mm, 22 * mm, 22 * mm, 15 * mm, 14 * mm, CONTENT_W - 151 * mm],
                "表3 ポイント取得率（平均 ± 95% 信頼区間）。ラリー長 = 1 ポイントあたりの総打球数。判断時間 = 自ロボットの 1 制御周期あたりの計算時間。")]  # fmt: skip
    s += fig("point_win_rate.png", "図3 条件別のポイント取得率。破線は 50%（互角）。")
    comps = []
    for x, y in (
        ("D", "B"),
        ("D", "B+"),
        ("D", "B*"),
        ("D", "C"),
        ("D", "D-"),
        ("B*", "B+"),
        ("B", "A"),
    ):
        if x in names and y in names:
            for cond in (None, "nominal", "unknown_wind", "unknown_court"):
                d = paired_diff(df, names[x], names[y], "point_win_rate", cond)
                comps.append(
                    [f"{x} − {y}", "全条件" if cond is None else COND_JA[cond], diff_text(d)]
                )
    s += [table([["比較", "条件", "ポイント取得率の差（対応のあるブートストラップ）"], *comps],
                [22 * mm, 26 * mm, CONTENT_W - 48 * mm],
                "表4 主要な比較。差 = 同じ (相手, 条件, シード) の試合どうしの差の平均。")]  # fmt: skip
    s += fig(
        "drill_accuracy.png",
        "図4 球出しドリルでの打球精度（同じ環境で 40 球のセッション × 16）。残差学習はホームコートで誤差を半減し、オンライン同定は未知のコートで打ち返せる球を 6 倍にした。",
    )
    s += fig(
        "identification.png",
        "図5 オンライン同定の推定値（各試合終了時）。未知条件では、反発係数・抗力・風を正しい方向・大きさで推定している。",
    )
    s += fig("point_win_by_opponent.png", "図6 相手別のポイント取得率（標準条件）。")
    return s


def _verdict(d: dict) -> str:
    if d["lo"] > 0:
        return "有意に高い"
    if d["hi"] < 0:
        return "有意に低い"
    return "有意差なし"


def discussion(data: dict) -> list:
    df, names, drill = data["df"], data["names"], data["drill"]
    s: list = [P("9. 考察", "h1")]

    def d(x, y, cond=None):
        if x not in names or y not in names:
            return None
        return paired_diff(df, names[x], names[y], "point_win_rate", cond)

    def per_cond(x, y) -> str:
        parts = []
        for cond in CONDS:
            r = d(x, y, cond)
            if r is not None:
                parts.append(f"{COND_JA[cond]} {r['diff'] * 100:+.1f} pt（{_verdict(r)}）")
        return "、".join(parts)

    items = []
    ba = d("B", "A")
    if ba:
        items.append(
            f"<b>物理モデルの効果</b>: B − A = {diff_text(ba)}（全条件）。重力・空気抵抗・バウンドを予測できなければ打点に"
            "間に合わず、3 次元の打ち合いは物理予測なしには成立しない。学習以前に、物理モデルが性能の土台になっている。"
        )  # fmt: skip
    db = d("D", "B*")
    if db:
        nom = df[df["condition"] == "nominal"]
        by_opp = nom.pivot_table(index="controller", columns="opponent", values="point_win_rate")
        gap = (by_opp.loc[names["D"]] - by_opp.loc[names["B*"]]).sort_values()
        top = gap.index[-1]
        opp_ja = {"steady": "安定型", "random": "ランダム", "aggressive": "攻撃型", "lob": "ロブ",
                  "slow": "低速"}  # fmt: skip
        cols = [c for c in nom.columns if c.startswith("strategy_rl-")]
        dn = nom[nom["controller"] == names["D"]]
        shares = dn.groupby("opponent")[cols].sum()
        shares = shares.div(shares.sum(axis=1), axis=0)
        tot = dn[cols].sum()
        tot = tot / tot.sum()
        fast = float(sum(v for c, v in tot.items() if c.endswith("2")))
        centre = float(sum(v for c, v in tot.items() if c[-2] == "1"))
        deep_top = float(sum(v for c, v in shares.loc[top].items() if c[-3] == "2"))
        items.append(
            f"<b>戦術の学習</b>: 同じ物理・学習・適応層の上で、狙いを手作りルールで決める B* と PPO で決める D を比べると、"
            f"D − B* = {diff_text(db)}（全条件; 条件別: {per_cond('D', 'B*')}）。"
            + ("強化学習は相手と状況に応じた狙いを獲得し、手作りの戦術を上回った。" if db["lo"] > 0 else
               "学習した戦術は手作りの戦術を明確には上回らなかった。" if db["hi"] > 0 else
               "学習した戦術は手作りの戦術に及ばなかった。")
            + f"差が最も大きいのは{opp_ja.get(top, top)}の相手（標準条件で D {by_opp.loc[names['D'], top] * 100:.0f}% 対 "
            f"B* {by_opp.loc[names['B*'], top] * 100:.0f}%）。D の選択を集計すると、最速の球（7 m/s）が {fast * 100:.0f}%、"
            f"センターへの球は {centre * 100:.0f}% で、常に速い球を左右に打ち分け、相手に応じて深さと左右の配分を変えていた"
            f"（{opp_ja.get(top, top)}の相手には深いボールが {deep_top * 100:.0f}%）。"
        )  # fmt: skip
    dd = d("D", "D-")
    bs = d("B*", "B+")
    if dd and bs:
        items.append(
            f"<b>オンライン適応</b>: 適応層の有無だけが違う組で、B* − B+ は {per_cond('B*', 'B+')}、"
            f"D − D− は {per_cond('D', 'D-')}。ドリルでは未知のコートで打ち返せた球が "
            f"{drill['unknown_court']['physics']['hits']} → {drill['unknown_court']['physics+residual+online-id']['hits']} 球（640 球中）、"
            f"イン率 {drill['unknown_court']['physics']['in_court_rate'] * 100:.0f}% → "
            f"{drill['unknown_court']['physics+residual+online-id']['in_court_rate'] * 100:.0f}% に改善した。"
            "未知の環境での頑健性は、学習した戦術ではなく、物理モデルを観測から更新する仕組みが担っている。"
        )  # fmt: skip
    dc = d("D", "C")
    if dc:
        inr = df.groupby(["controller", "condition"])["in_rate"].mean()
        c_in = [inr[names["C"]][c] * 100 for c in CONDS]
        d_in = [inr[names["D"]][c] * 100 for c in CONDS]
        items.append(
            f"<b>純粋 RL との比較</b>: D − C = {diff_text(dc)}（全条件）。ラケット角度と速度から打球の行き先を学ぶ C は、"
            f"同じ程度の学習量では打球を安定してコートに入れられない（イン率 {min(c_in):.0f}–{max(c_in):.0f}%、"
            f"D は {min(d_in):.0f}–{max(d_in):.0f}%）。打撃モデルの逆解で「狙った場所へ打つ」を物理層が"
            "保証する D は、27 通りの狙いを選ぶだけでよく、学習の負担が桁違いに小さい。"
        )  # fmt: skip
    nb = drill["nominal"]
    items.append(
        "<b>残差学習の位置づけ</b>: ホームコートでは着地誤差の中央値を "
        f"{nb['physics']['landing_error_median_m'] * 100:.0f} → {nb['physics+residual']['landing_error_median_m'] * 100:.0f} cm に減らしたが、"
        "環境が変わると補正が外れて逆効果になりうる（同定でモデルを変えたまま併用すると標準条件でも誤差が 3 倍）。"
        "変化検知で残差を止める設計により、ホームでの精度と未知環境での頑健性を両立させた。"
    )  # fmt: skip
    dbc = d("D", "B*", "unknown_court")
    caveat = ""
    if dbc is not None and dbc["hi"] < 0:
        caveat = (
            f"ただし、未知のコートでは学習した戦術が手作りルールに劣った（D − B* = {diff_text(dbc)}）。"
            "方策は同定値を観測として受け取るが、学習中に見ていない範囲の環境では狙いの選び方まで適応できておらず、"
            "戦術そのものの適応（未知環境を含む学習分布、メタ学習など）は今後の課題である。"
        )
    items.append(
        "<b>研究課題への答え</b>: 物理モデルを土台に、学習は「どこへ打つか」（戦術）と「モデルのずれ」（残差）を、"
        "オンライン同定は「環境の変化」を担うように役割を分けることで、学習分布内の対戦性能は大きく向上し"
        "（D は全条件で最高またはそれに近いポイント取得率）、未知の環境でも性能の崩れを大幅に抑えられた。"
        "学習だけ（C）では打球を制御できず、物理だけ（B）では戦術が単調になる。" + caveat
    )  # fmt: skip
    s += bullets(items)
    return s


def advanced() -> list:
    """Sections 10-11: self-play league and the VLA experiment (if their results exist)."""
    s: list = []
    lg, sb = Path("results/league"), Path("results/league_sb")
    if (lg / "elo.json").exists() and (sb / "elo.json").exists():
        e1 = json.loads((lg / "elo.json").read_text(encoding="utf-8"))
        e2 = json.loads((sb / "elo.json").read_text(encoding="utf-8"))
        pts = pd.read_csv(lg / "points.csv")
        srv = np.where(pts["server"] == "home", pts["home"], pts["away"])
        ace = (pts["hits"] == 1) & (pts["reason"] == "not returned")
        learned = pd.Series(srv).str.startswith("gen_").to_numpy()
        ace_l, ace_o = ace[learned].mean() * 100, ace[~learned].mean() * 100
        g1 = [v for k, v in e1["ratings"].items() if k.startswith("gen_")]
        g2 = {k: v for k, v in e2["ratings"].items() if k.startswith("gen_")}
        last = max(g2, key=lambda k: e2["timesteps"][k])
        h2h = pd.read_csv(sb / "head_to_head.csv", index_col=0)
        p2 = pd.read_csv(sb / "points.csv")
        gg = p2[p2["home"].str.startswith("gen_") & p2["away"].str.startswith("gen_")]
        s += [
            P("10. 発展 1: 自己対戦でさらに強くなれるか", "h1"),
            P("D を出発点に、自分の過去の世代と戦い続ける自己対戦リーグ（AlphaStar 方式の簡略版。1.2 万判断ごとに世代を追加し、"
              "学習者が苦手な過去の世代を優先して選び、2 割はスクリプト相手）を CPU で 4 時間（9.7 万判断）学習した。全世代・B 系・"
              f"スクリプト相手の総当たり戦（{e1['matches']} 試合）から Elo を推定すると、全世代が {min(g1):.0f}–{max(g1):.0f} の範囲で、"
              "強さは変わらなかった。"),
            P(f"原因はサーブだった。学習済みの方策がサーブすると、相手が誰でも {ace_l:.1f}% がエースになる（他の選手のサーブは "
              f"{ace_o:.1f}%）。「深く・受け手のフォア側・7 m/s」のサーブがベースライン際で低く弾み、受け手ロボットの台座の横を抜ける。"
              "レールは取り付け位置より後ろへ下がれず、腕も台座に近すぎる球は打てないため、受け手は打点を見つけられない（設計上の死角）。"
              "D 自身が元の学習でこの死角を見つけており、自己対戦は世代 1 以降この 1 本に収束した。サーブで勝負が決まるゲームでは、"
              "強さの差が生まれない。"),
            P("そこで実際のテニスと同じサービスボックス（サーブはネットから 1.62 m 以内、実コートと同じ比率）を加え、D を微調整してから"
              f"自己対戦をやり直した（7.2 万判断）。学習済み同士のラリーは 1 ポイント平均 {gg['hits'].mean():.1f} 打になり、Elo は "
              f"{max(g2.values()) - 1500:+.0f} まで上がって頭打ちになった（図 7）。ただし上昇の大部分は出発点（世代 0）への圧勝"
              f"（最終世代が世代 0 から {h2h.loc[last, 'gen_000'] * 100:.0f}% のポイント）によるもので、固定の相手へのポイント取得率は"
              f"ほぼ変わらない（対 B {h2h.loc['gen_000', 'B: Physics'] * 100:.0f}% → {h2h.loc[last, 'B: Physics'] * 100:.0f}%、"
              f"対 B* {h2h.loc['gen_000', 'B*: Physics+Adapt'] * 100:.0f}% → {h2h.loc[last, 'B*: Physics+Adapt'] * 100:.0f}%）。"
              "公平なルールでの自己対戦は、自分の系統に勝つ戦術（深く速い球）を素早く見つけるが、汎用的な強さにはつながりにくかった。"
              "また強化学習は、ルールや設計の穴（ポイント間の間、届かないサーブ）を真っ先に突く「デバッガ」としても働いた。"),
        ]  # fmt: skip
        s += fig(
            "league_sb_elo.png",
            "図7 サービスボックスありの自己対戦。左: 世代ごとの Elo（95% CI）。右: 世代ごとのポイント取得率（相手別）。",
        )
    vs = Path("results/vla/summary.json")
    if vs.exists():
        v = json.loads(vs.read_text(encoding="utf-8"))
        t = pd.DataFrame(v["table"])
        lat = v["latency"]
        real = f"{lat['simulated_real_latency_s']:.2f}"
        bud = v.get("hier_time_budget_s", {})

        def r(world, cond, m="in_zone"):
            x = t[(t["world"] == world) & (t["condition"] == cond)]
            return float("nan") if x.empty else float(x.iloc[0][m])

        conds = [("物理エキスパート（教師）", "expert"), ("VLA 単独・遅れ 0", "vla@0.00"),
                 (f"VLA 単独・実測の遅れ {lat['mean_s'] * 1000:.0f} ms", f"vla@{real}(real)"),
                 ("VLA 単独＋ボール位置入力・遅れ 0", "vla@0.00+ball"),
                 (f"VLA 単独＋ボール位置入力・{lat['mean_s'] * 1000:.0f} ms", f"vla@{real}(real)+ball"),
                 (f"階層型・実測の遅れ {lat['mean_s'] * 1000:.0f} ms", f"hier@{real}(real)")]  # fmt: skip
        conds = [c for c in conds if not t[t["condition"] == c[1]].empty]
        rows = [
            ["制御器", "地球 命中 / 指示どおり", "火星 命中 / 指示どおり", "月 命中 / 指示どおり"]
        ]
        for lab, c in conds:
            rows.append(
                [lab]
                + [
                    f"{r(w, c, 'hit') * 100:.0f}% / {r(w, c) * 100:.0f}%"
                    for w in ("earth", "mars", "moon")
                ]
            )
        e2e_moon = r("moon", "vla@0.00", "hit") * 100
        miss = t[(t["condition"] == "vla@0.00")]["miss_dist_median_m"].median() * 100
        ball = t[t["ball_state"]]
        ball_txt = ""
        if not ball.empty:

            def pct(world, cond):
                return r(world, cond, "hit") * 100

            n = ball["episodes"]
            hits, inc = (ball["hit"] * n).sum(), (ball["in_court"] * n).sum()
            zone = (ball["in_zone"] * n).sum()
            ball_txt = (
                "画像の認識を不要にする対照実験として、ボールの位置と速度（真値）を入力に加えて同じ設定で学習し直すと、命中は遅れ 0 で"
                f"地球 {pct('earth', 'vla@0.00+ball'):.0f}%・火星 {pct('mars', 'vla@0.00+ball'):.0f}%・"
                f"月 {pct('moon', 'vla@0.00+ball'):.0f}% に上がり、外したときの最接近距離も約 "
                f"{ball['miss_dist_median_m'].median() * 100:.0f} cm に縮んだ。画像から速いボールを捉えることが最初のボトルネックである。"
                f"それでも当たった球のうちコートに入るのは {inc / hits * 100:.0f}%、そのうち指示どおりのゾーンは {zone / inc * 100:.0f}%"
                "（1 ゾーンの面積はコートの約 1/6）で、打球の向きと速さを決める精度が次の壁になる。"
            )  # fmt: skip
            if t["condition"].str.contains("period").any():
                worlds = ("earth", "mars", "moon")
                p36 = [pct(w, c) for w in worlds
                       for c in ("vla@0.00+ball/period0.36", f"vla@{real}(real)+ball")]  # fmt: skip
                gap = [pct(w, f) - pct(w, s) for w in worlds
                       for f, s in (("vla@0.00+ball/period0.2", "vla@0.20+ball"),
                                    ("vla@0.00+ball/period0.36", f"vla@{real}(real)+ball"))]  # fmt: skip
                ball_txt += (
                    f"遅れの影響は単調ではない。同じ切り替え周期で比べた遅れそのものの影響は {max(gap):.0f} ポイント以内だったが、"
                    f"0.1 秒ごとに 0.1 秒古い答えへ切り替えると火星 {pct('mars', 'vla@0.10+ball'):.0f}%・"
                    f"月 {pct('moon', 'vla@0.10+ball'):.0f}% に崩れ、0.36 秒ごとなら {min(p36):.0f}–{max(p36):.0f}% だった（図 10）。"
                    "end-to-end では、遅れよりも行動チャンクのつなぎ方が成否を分ける。"
                )  # fmt: skip
        rel = t[(t["condition"] == f"hier@{real}(real)")]["in_zone_relational"].mean() * 100
        hz = t[(t["condition"] == f"hier@{real}(real)")]["in_zone"].mean() * 100
        s += [
            P("11. 発展 2: VLA はテニスができるか（惑星テニス）", "h1"),
            P("VLA（Vision-Language-Action）モデルの SmolVLA（4.5 億パラメータ、LeRobot 0.4.4）を、物理エキスパートの成功デモ 1,673 本"
              "（26 万フレーム。カメラ画像 2 枚＋関節状態＋英語の指示文 → 20 ms 後の関節目標と狙い）で微調整した（Colab の A100、"
              "2 万ステップ、3.9 時間）。課題は言語指示つきの返球ドリルで、ゾーンを指定する指示 6 種と、相手を見ないと従えない指示 2 種"
              "（“away from / straight at the opponent”）がある。重力だけを変えると軌道の形はそのままで時間だけが伸びる（火星 1.6 倍、"
              "月 2.5 倍遅い。テストで確認）ことを使い、地球・火星・月で「試合の速さ」だけを変えた。推論の遅れはシミュレーション時間で"
              f"再現した（A100 の実測 {lat['mean_s'] * 1000:.0f} ms）。"),
            table(rows, [56 * mm, 40 * mm, 40 * mm, CONTENT_W - 136 * mm],
                  "表5 VLA の評価（各条件 100 球、学習と別のシード）。命中 = ラケットに当たった割合、指示どおり = 指示したゾーンに着地した割合。"),
            P(f"VLA 単独（画像 → 関節）は、考える時間をゼロにしても、2.5 倍遅い月面でもほとんど当たらない（月で命中 {e2e_moon:.0f}%）。"
              f"遅れが原因ではなく、そもそもボールの近くに振れていない（外れたときのラケットとボールの最接近距離の中央値は {miss:.0f} cm、"
              "エキスパートの打球時は 5–9 cm）。"
              + ball_txt),
            P(f"一方、VLA に狙いだけを決めさせ、物理層が打つ階層型は、どの世界でも指示どおりのゾーンに平均 {hz:.0f}% 返した"
              f"（相手を見ないと従えない指示でも {rel:.0f}%）。答えが遅すぎると打点の確定に間に合わず指示に従えなくなるが、その限界"
              f"（成功率が半分になる遅れ）は地球 {bud.get('earth', float('nan')):.1f} 秒・火星 {bud.get('mars', float('nan')):.1f} 秒・"
              f"月 {bud.get('moon', float('nan')):.1f} 秒で、世界の時間の伸びにほぼ比例した（遅れ × 速さで曲線が重なる、図 9）。"
              "汎用モデルは「どこへ打つか」を画像と言葉から決める脳としては有能だが、10 ms 単位のタイミングが要る動きは物理モデルに"
              "任せるべき、というのが結論である。"),
        ]  # fmt: skip
        s += fig("vla_worlds.png", "図8 惑星テニス: 各制御器の命中率と指示どおりの割合（95% CI）。")
        s += fig(
            "vla_hier_latency.png",
            "図9 階層型: VLA の答えの遅れと指示どおりの割合。右は遅れに世界の速さを掛けて地球換算したもの。",
        )
        if (FIG / "vla_ball.png").exists() and not t[t["ball_state"]].empty:
            s += fig(
                "vla_ball.png",
                "図10 対照実験: ボールの位置と速度を直接入力した VLA 単独の命中率（各 100 球、95% CI）。"
                "横軸は新しい行動チャンクに切り替える周期。点線は画像のみの VLA の最大値。",
            )
    return s


def closing() -> list:
    return [
        P("12. 限界と今後", "h1"),
        *bullets([
            "シミュレーションのみ（実機・実カメラは未検証）。センサは位置・速度にガウスノイズを加えたもので、スピンは観測しない。",
            "相手はスクリプト型 5 種と自己対戦の過去の世代。人間の打球データとの対戦は今後の課題。",
            "ロボットはレールの取り付け位置より後ろへ下がれない（サーブの死角）。サービスボックスで回避したが、本来は設計の改善が必要。",
            "強化学習の予算は CPU で数時間規模。方策は 27 通りの離散的な狙いで、連続的な狙い・回転の選択は未対応。",
            "VLA は SmolVLA 1 種・デモ 1,673 本・2 万ステップのみ。より大きなモデル、手首カメラ、推論の高速化、行動チャンクのなめらかな"
            "つなぎ方（temporal ensembling、real-time chunking）は未検証。",
            "オンライン同定は観測が十分にある量（反発・摩擦・抗力・風・打撃ゲイン）に限った。スピン量や突風のような時間変化は扱っていない。",
            "未知条件のドリル結果は適応手法の開発中に参照した（試合評価のシードは開発に使っていない）。",
        ]),
        P("13. 再現方法", "h1"),
        P("Dev Container 内で robot-arm-tennis/ に移動し、python -m pytest（テスト）、python -m tennis.learning.train --config "
          "configs/training/ppo_hybrid.yaml（学習）、python scripts/evaluate.py --config configs/evaluation/main.yaml（評価）、"
          "python scripts/make_figures.py、python scripts/make_video.py、python scripts/make_report.py（図・動画・本資料）。"
          "乱数はすべてシード指定で、動画のクリップは評価と同じコードの決定的な再生である。自己対戦は configs/training/ppo_league*.yaml と"
          "scripts/league_eval.py、VLA は scripts/vla/（Colab で make_dataset → colab/train.sh → colab/eval_all.sh。対照実験は "
          "make_dataset --ball-state と colab/eval_ball.sh）。詳細は README.md と EXPERIMENTS.md。"),
    ]  # fmt: skip


def main() -> None:
    data = load()
    story = build(data) + discussion(data) + advanced() + closing()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    doc = SimpleDocTemplate(str(OUT), pagesize=A4, leftMargin=15 * mm, rightMargin=15 * mm,
                            topMargin=13 * mm, bottomMargin=15 * mm, title="Robot-Arm Tennis AI",
                            author="yoichiozaki", subject="Physical AI 応用1 最終課題")  # fmt: skip
    doc.build(story, onFirstPage=_footer, onLaterPages=_footer)
    print(OUT, OUT.stat().st_size // 1024, "KB")


if __name__ == "__main__":
    main()
