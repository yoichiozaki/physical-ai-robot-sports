# STATUS.md — robot-arm-tennis

最終更新: 2026-10-10

## 状態: ✅ 本編・発展編（自己対戦・VLA・対照実験）ともに完了

| 項目 | 状態 | 根拠 |
|---|---|---|
| シミュレーション（コート・空力ボール・8 自由度ロボット ×2・審判・ポイント間の間・任意のサービスボックス） | ✅ 検証済み | `tests/physics`, `tests/unit`, `tests/integration` |
| 物理予測制御・残差学習・オンライン同定 | ✅ 検証済み | EXPERIMENTS E1–E4 |
| 強化学習（D: Hybrid / C: 純 RL） | ✅ | `checkpoints/ppo_*/run_info.json` |
| 本編の評価（7 制御器 × 相手 5 × 条件 4 × シード 12 = 1,680 試合） | ✅ | `results/eval_main/`, EXPERIMENTS E7 |
| 発展1: 自己対戦リーグ（元のルール / サービスボックスあり）・攻略役・総当たり Elo | ✅ | `results/league*/`, EXPERIMENTS E8–E9 |
| 発展2: 惑星テニス × SmolVLA（Colab A100、デモ 1,673 本、評価 3,660 エピソード） | ✅ | `results/vla/`, EXPERIMENTS E10 |
| 発展2 対照実験: ボールの位置・速度を入力した VLA（再学習、評価 1,800 エピソード） | ✅ | `results/vla/ball*/`, EXPERIMENTS E10b |
| 図・動画（173.5 秒）・説明資料 PDF（8 ページ） | ✅ 最終版 | `results/figures/`, `videos/`, `reports/` |
| テスト・lint | ✅ 34 テスト合格、ruff クリーン（2026-10-10 再実行） | `python -m pytest -q`, `ruff check .` |

## Colab の使用

- A100（5.30 CU/時）× 最大 2 台。データ生成・SmolVLA 学習 2 回・評価・クリップ描画で約 72 CU（残高 199.05 → 127.16）。
  すべてのランタイムは停止済み。
- 学習済み SmolVLA（画像のみ、865 MB）はローカルの `checkpoints/smolvla_tennis/`（Git 管理外）に保存。ボール入力のモデルと
  データセット（約 2 GB）は `python scripts/vla/make_dataset.py`（`--ball-state`）と `colab/train.sh` で再生成できるので保存していない。

## Next commands（再現・更新する場合）

```bash
cd robot-arm-tennis
python -m pytest -q
python scripts/league_eval.py --league checkpoints/ppo_league_sb --service-line 1.62 --out results/league_sb
python scripts/vla/summarize_vla.py && python scripts/make_figures.py
python scripts/make_video.py && python scripts/make_report.py
```

## 既知の限界

- 未知: コートでは学習した戦術（D）が手作りルール + 適応（B*）に 5 pt 劣る（E7）。
- ロボットはレールの取り付け位置より後ろへ下がれない（深いサーブの死角、E8）。サービスボックスで回避（T-025）。
- 自己対戦の強化は主に自分の系統に対するもの（E9）。VLA 単独では返球できない（E10）。ボールの位置を直接与えると半分ほど
  当たるが、狙ったゾーンにはほとんど返せず、行動チャンクの切り替え方に敏感（E10b）。
