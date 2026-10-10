# checkpoints/

学習の出力先（`python scripts/train.py --out checkpoints`）。各ランのディレクトリ構成:

| ファイル | 内容 | Git |
|---|---|---|
| `final_model.zip` | 学習終了時の SB3 PPO モデル（評価・動画で使用） | 管理する（数 MB） |
| `run_info.json` | アルゴリズム、シード、ハイパーパラメータ、総ステップ、学習時間、バージョン、git commit | 管理する |
| `best_model.zip` | 学習中の評価で最良だったモデル | 管理しない |
| `checkpoints/ckpt_<N>_steps.zip` | 定期チェックポイント（`--resume` で再開） | 管理しない |
| `progress.csv`, `eval/` | 学習ログ・評価ログ（学習曲線の元データ） | `progress.csv` は管理する |

使用モデル:
- `ppo_hybrid_s0/final_model.zip` — D: Hybrid AI（戦略選択、イベント駆動判断）
- `ppo_direct_s0/final_model.zip` — C: 純粋な強化学習（目標位置を直接出力）
