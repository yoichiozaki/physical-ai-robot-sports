# STATUS.md — 現在の状態

最終更新: 2026-10-10 — **全フェーズ完了**

## 状態の凡例
✅ 実装済み・検証済み / 🟡 実装済み・一部未検証

## フェーズ状態

| Phase | 内容 | 状態 | 証拠 |
|---|---|---|---|
| 0 | プロジェクトメモリ | ✅ | AGENTS / PLAN / STATUS / DECISIONS / EXPERIMENTS |
| 1 | Dev Container | ✅ | 公式 devcontainer CLI 0.89 で `up` 成功・postCreate の doctor OK、EGL/OSMesa 両方で描画確認 |
| 2 | 物理シミュレーション | ✅ | `tests/physics`（9 件）合格、反発係数校正（EXP-001）、打ち返し動画 `videos/clips/strike.mp4` 相当を動画 04 に収録 |
| 3 | 物理予測制御 | ✅ | 打撃方向誤差 0.12°（60/60 打撃成功）、予測誤差 2.1 cm@0.5 s、防御ドリル 100%/94%（`results/benchmark.json`） |
| 4 | Gymnasium 環境 | ✅ | `check_env` 合格（direct/hybrid）、境界・NaN・終了・シード再現・イベント駆動判断のテスト |
| 5 | 強化学習 (PPO) | ✅ | `checkpoints/ppo_hybrid_s0`, `checkpoints/ppo_direct_s0`（final_model, run_info, progress） |
| 6 | Colab 連携 | ✅ | CLI 0.7.4 の全コマンド確認、承認ゲート付き計画/実行、Drive 再開、ノートブック、ローカル smoke test 合格。OAuth 認証済み（2026-10-09）。本プロジェクトの学習・評価はすべてローカル CPU で実行（CU 使用 0） |
| 7 | 追加機能 | ✅ | 戦略選択（Hybrid）、相手 5 種、DR、オンライン同定、計測フィルタ（評価で効果確認） |
| 8 | 評価実験 | ✅ | 2,800 試合（`results/eval_main/`）、対応のある bootstrap CI（`results/summary.json`）、図 11 枚 |
| 9 | 可視化 | ✅ | 3D オーバーレイ（予測軌道・反射点・打撃点・狙い・目標）+ 日本語 HUD |
| 10 | デモ動画 | ✅ | `videos/PAI最終課題_yoichiozaki.mp4`（148.0 s, H.264/yuv420p, 1280×720, 30 fps, 7.9 MB、全フレーム確認） |
| 11 | PDF レポート | ✅ | `reports/説明資料_PAI最終課題_yoichiozaki.pdf`（6 ページ, IPAex フォント埋め込み、全ページ目視確認） |
| 12 | リポジトリ | ✅ | README・LICENSE・CI・再現手順。個人情報・秘密情報なし |

## 主要な結果（`results/eval_main`、標準条件、相手 5 種 × 20 試合）
- 勝率: A 33% / B 80% / B+ 91% / B* 80% / C 97% / **D 100%** / **D* 100%**
- 得点/60 s: B 7.26 → D 9.20（D−B 得失点差 +1.77 [95% CI +0.98, +2.54]）
- 守備型相手: B 0 点 → D 5.65 点・全勝
- 未知条件の勝率（高摩擦 / 高反発）: B 71/79%、D 70/78%、D* 82/82%（予測誤差 約半減）

## 成功済みテスト（2026-10-10 再実行）
- `pytest`: 36 passed（slow・render 含む）
- `ruff check .` / `ruff format --check .`: OK（リポジトリ全体 142 ファイル）

## 既知の制約
- 学習済み方策は各 1 シード（評価は 20 試合 × 条件）。
- 予測モデルはゴールポスト角を近似していない（0.5 s 先 p95 誤差 13.8 cm の主因）。
- `progress.csv` は学習ログから再構成（D-019）。

## Next commands
```powershell
# 再現: 評価 → 図 → 動画 → レポート
docker exec pai-dev bash -c "cd air-hockey && python scripts/evaluate.py --config configs/evaluation/main.yaml && python scripts/summarize_results.py && python scripts/make_figures.py && python scripts/make_video.py && python scripts/make_report.py"
# テストと lint
docker exec pai-dev bash -c "cd air-hockey && python -m pytest -q && ruff check ."
```
