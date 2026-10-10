# PLAN.md — 全体計画

- 締切: **2026-11-02 10:00 JST**（作業開始: 2026-10-09）
- 優先順位: ①動作するロボット ②実証済み AI 機能 ③比較評価 ④動画と PDF ⑤追加機能
- 完成保証ライン: Phase 3（物理予測制御）。AI 機能（Phase 5/7）は必ず追加する。

## マイルストーン（実日付で更新）

| 日付 | 目標 | 状態 |
|---|---|---|
| 2026-10-12 | 物理シミュレーション基本動作（ロボットがパックを打ち返す動画） | ✅ 2026-10-09 達成 |
| 2026-10-18 | 物理予測制御の完成 | ✅ 2026-10-09 達成 |
| 2026-10-24 | 強化学習・初期評価 | ✅ 2026-10-09 達成 |
| 2026-10-28 | 比較実験とデモ完成 | ✅ 2026-10-09 達成 |
| 2026-10-31 | PDF・動画・リポジトリ完成 | ✅ 2026-10-09 達成 |

## フェーズ一覧

依存関係: P1 → P2 → P3 → P4 → P5 → (P6, P7) → P8 → (P9, P10, P11) → P12

### Phase 1: Development Foundation
- [x] Dev Container（Dockerfile, devcontainer.json）、バージョン固定の依存関係
- [x] pytest / Ruff 設定、`.gitignore`
- [x] MuJoCo ヘッドレス実行・オフスクリーン描画・動画出力の検証
- 完了条件: イメージのビルド成功 / import テスト成功 / MuJoCo 実行 / オフスクリーン描画 / pytest 実行 / セットアップ手順

### Phase 2: Physical Simulation
- [x] MJCF 生成（テーブル・壁・ゴール・パック・ロボット(2軸ガントリー)・相手ロボット）
- [x] 関節・アクチュエータ・位置/速度/加速度制限、衝突、摩擦（粘性減衰）、壁反発、ゴール判定、スコア、リセット
- [x] 物理テスト 8 種（静止安定・壁反射・マレット衝突・貫通防止・可動範囲・数値安定・ゴール判定・再現性）
- 完了条件: ロボットがパックを打ち返す動画の保存

### Phase 3: Physics-Based Control（完成保証ライン）
- [x] Reactive / Physics Prediction（壁反射込み）/ Interception Planner / Attack Planner
- [x] 到達時間 vs ロボット移動時間の比較、到達不能打点の除外
- [x] 評価: 予測誤差・迎撃誤差・防御成功率・制御レイテンシ、予測軌道の可視化
- 完了条件: 単純攻撃を防御、防御成功率の自動測定、Reactive vs Physics 比較

### Phase 4: Gymnasium Environment
- [x] reset/step/spaces/render/close、状態ベクトル観測、連続行動（目標位置）
- [x] 報酬（YAML で重み変更可能）
- [x] テスト: env checker / 形状 / 行動範囲 / NaN・Inf なし / 終了 / シード再現性
- 完了条件: ランダム・ルールベース方策で複数エピソード実行

### Phase 5: Reinforcement Learning（PPO, SB3, MlpPolicy）
- [x] 学習ループ、VecEnv、評価/チェックポイント callback、ログ、ハイパラ設定、決定的評価、再開
- [x] カリキュラム Stage 1–6
- [x] Pure RL（目標位置を直接出力）と Hybrid（物理プリミティブ上の戦略選択）
- 完了条件: 学習実行・チェックポイント作成・再ロード・再現可能評価・CSV/JSON 出力

### Phase 6: Colab Integration
- [x] 公式情報で Colab CLI の仕様を確認（推測コマンド禁止）
- [x] ローカルと共通コードでのリモート実行手順（ノートブック/スクリプト）、Drive へのチェックポイント永続化、再開
- [x] 認証・CU 利用（OAuth 認証済み。ホッケーの学習はローカル CPU のみで CU 使用 0）
- 完了条件: ローカル smoke test 成功、リモート手順の文書化

### Phase 7: Advanced Features（効果の高いものを選択）
- [x] P1 Adaptive Strategy Selection（Defense / Direct / Bank）
- [x] P2 相手戦略 5 種（Stationary / Random / Reactive / Defensive / Aggressive）
- [x] P3 Domain Randomization（摩擦・反発・初速・相手速度・観測ノイズ）
- [x] P4 Hybrid Physics + Learning
- [x] P5 Residual / 適応的予測補正（時間があれば）

### Phase 8: Experimental Evaluation
- [x] A Reactive / B Physics / C Learning / D Hybrid（+ B+ / B* / D* の計 7 制御器）
- [x] 勝率・得点・失点・防御成功率・ラリー長・予測誤差・判断レイテンシ・汎化性能
- [x] 評価シード複数（2,800 試合）、対応のある bootstrap CI、未知条件、実行時間、CSV/JSON、グラフ再生成スクリプト
      （学習シードは各方策 1 本）

### Phase 9: Visualization
- [x] テーブル・色分け・スコアボード・予測軌道・迎撃点・目標位置・制御器名・指標・戦略表示（実アルゴリズム出力のみ）

### Phase 10: Demo Video
- [x] `videos/PAI最終課題_yoichiozaki.mp4`（148.0 s、1280×720 30 fps、H.264/yuv420p、倍速表記）

### Phase 11: Final Report
- [x] `reports/説明資料_PAI最終課題_yoichiozaki.pdf`（日本語 6 ページ、IPAex フォント埋め込み、全ページ目視確認）

### Phase 12: Repository Completion
- [x] README、CI、テスト成功、秘密情報・個人情報なし、ライセンス、再現手順
