# Physical AI Robot Sports

Physical AI 応用1（OmniCampus: `yoichiozaki`）のための、物理シミュレーション上のロボット対戦 AI 2 題。
どちらも同じ研究課題 —「物理モデルによる予測制御に、機械学習（戦術・補正）とオンライン適応を組み合わせると、
対戦性能と未知環境への適応は改善するか？」— を、難易度の違う題材で検証する。

| プロジェクト | 題材 | ロボット | 状態 |
|---|---|---|---|
| [air-hockey/](air-hockey/README.md) | 2D エアホッケー（壁反射・減衰） | 2 軸ガントリー ×2 | 完成（動画・PDF・評価・テスト） |
| [robot-arm-tennis/](robot-arm-tennis/README.md) | 3D ミニテニス（空気抵抗・マグヌス・バウンド・ネット） | 床ガントリー 2 軸 + 6 軸アーム（8 自由度）×2 | 完成（動画・PDF・評価・テスト） |

## 共通の開発環境

すべて Dev Container 内で実行する（ホストへの Python 導入は不要）。

```powershell
docker build -t pai-dev -f .devcontainer/Dockerfile .
docker run -d --name pai-dev --shm-size=2g -v "${PWD}:/workspaces/physical-ai-robot-sports" pai-dev sleep infinity
docker exec pai-dev bash -c "pip install --no-deps -e air-hockey -e robot-arm-tennis"
docker exec pai-dev bash -c "cd robot-arm-tennis && python -m pytest -q"
```

VS Code では「Dev Containers: Reopen in Container」。依存関係は `requirements/lock.txt` に固定、
CI（`.github/workflows/ci.yml`）は両プロジェクトの `ruff` と `pytest` を実行する。
開発規則は [AGENTS.md](AGENTS.md)、全体の状態は [STATUS.md](STATUS.md)。

## ライセンス

[MIT](LICENSE)。ロボット・コート・テーブルのモデルはすべてコードから手続き生成しており、外部アセットは使っていない。
