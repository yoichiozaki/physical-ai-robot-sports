# checkpoints/

| ディレクトリ | 内容 | 再生成 |
|---|---|---|
| `residual/model.pt` (+ `model.json`) | 着地点の残差モデル（MLP 16→64→64→2、ホームコートのドリル 4,575 球で学習） | `python scripts/residual.py collect --condition nominal --tag nominal` → `train` |
| `ppo_hybrid/` | D: Hybrid AI（PPO、MultiDiscrete 3×3×3 = コース・深さ・球速） | `python -m tennis.learning.train --config configs/training/ppo_hybrid.yaml` |
| `ppo_direct/` | C: PPO direct（PPO、Box(3) = ラケット面の仰角・方位角・ラケット速度） | `python -m tennis.learning.train --config configs/training/ppo_direct.yaml` |

各 PPO ラン: `best_model.zip`（評価で最良、デプロイに使用）、`final_model.zip`、`progress.csv`（学習曲線）、
`run_info.json`（判断回数・所要時間・バージョン・git commit・設定）、`config.json`。途中のチェックポイント
（`checkpoints/ckpt_*.zip`）と評価ログ（`eval/evaluations.npz`）は Git 管理外。

学習の単位は「判断（= 自ロボットのショット 1 回）」。1 判断はおよそ 1.5–2 秒のシミュレーション
（物理 1 kHz、制御 50 Hz）に相当する。
