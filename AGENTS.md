# AGENTS.md — エージェント共通の開発規則

このリポジトリ（`physical-ai-robot-sports`）は Physical AI 応用1 の 2 つのプロジェクトを含む。

| ディレクトリ | 内容 | Python パッケージ |
|---|---|---|
| [`air-hockey/`](air-hockey/) | Adaptive Air Hockey AI（2D、2 軸ガントリーロボット）— 完成 | `hockey` |
| [`robot-arm-tennis/`](robot-arm-tennis/) | Robot-Arm Tennis AI（3D、ガントリー + 6 軸アーム同士のテニス）— 完成 | `tennis` |

共有: `.devcontainer/`（開発環境）、`requirements/`（固定した依存関係）、`.github/workflows/`（CI）、Colab CLI。

## 0. セッション開始時に読むファイル

1. この `AGENTS.md`
2. ルートの [`STATUS.md`](STATUS.md)（両プロジェクトの状態と共有インフラ）
3. 作業するプロジェクトの `PLAN.md` → `STATUS.md`（**Next commands** から再開）→ `DECISIONS.md` → `EXPERIMENTS.md`

## 1. 実行環境

- Python・ライブラリ・CLI はすべて Dev Container 内で実行する。Windows ホストへの Python 導入は要求しない。
- VS Code: 「Dev Containers: Reopen in Container」（ワークスペースは `/workspaces/physical-ai-robot-sports`）。
- エージェント/CLI から（VS Code を使わない場合）:
  ```powershell
  docker build -t pai-dev -f .devcontainer/Dockerfile .
  docker run -d --name pai-dev --shm-size=2g -v "${PWD}:/workspaces/physical-ai-robot-sports" -v pai-hockey-colab-cli:/home/vscode/.config/colab-cli pai-dev sleep infinity
  docker exec pai-dev bash -c "pip install --no-deps -e air-hockey -e robot-arm-tennis"
  docker exec pai-dev bash -c "cd robot-arm-tennis && python -m pytest -q"
  ```
- コマンドは各プロジェクトのディレクトリで実行する（`configs/`, `results/` などの相対パスはプロジェクト基準）。
- ヘッドレス描画は `MUJOCO_GL=egl`（失敗時 `osmesa`）。PPO の MlpPolicy は CPU が速いため学習は CPU を既定とする。
- Colab の OAuth トークンは Docker ボリューム `pai-hockey-colab-cli`（コンテナ内 `~/.config/colab-cli`）にあり、イメージ・Git には入らない。

## 2. コーディング規約

- 各プロジェクトは src レイアウト。Python 3.11、公開関数に型ヒント。プロジェクト間で import しない（各プロジェクトは単独で再現できること）。
- 物理パラメータ・報酬重み・ハイパーパラメータは `configs/*.yaml` または設定 dataclass に置く。
- ロボットは MuJoCo の関節とアクチュエータで動かす。`qpos` の直接書き換えはリセット時の初期化のみ。
- 例外を握りつぶさない。乱数は必ずシードを受け取る（`numpy.random.Generator`）。
- Lint: `ruff check .` / `ruff format --check .`。テスト: `pytest`（プロジェクトごと、CI は両方を実行）。

## 3. 品質ゲート（各フェーズ）

Implementation / Execution / Testing / Evidence（`results/`, `videos/`, ログ）/ Documentation の 5 条件を満たすまで完了扱いにしない。
テストの無効化・結果の捏造・演出目的の偽予測描画は禁止。状態は「実装済み / 検証済み / 未検証」を区別して記録する。

## 4. 外部サービス・破壊的操作

- 外部サービスの認証情報や有料クラウド資源を成果物に含めない。
- GitHub、YouTube、OmniCampus など外部サービスへの操作は、各サービスの現在の設定と利用者の意図を確認してから実行する。
- 既存データを破壊的に削除しない。

## 5. 個人情報・秘密情報

- 成果物（コード、ログ、PDF、動画、コミット）に氏名・所属・メールアドレスを含めない。記載してよい識別子は OmniCampus アカウント名 `yoichiozaki` のみ。
- コミット作成者は repo-local 設定 `yoichiozaki <26167202+yoichiozaki@users.noreply.github.com>`。
- 認証情報・ホストの絶対パスをファイルに書き込まない。

## 6. 成果物の場所（各プロジェクト内）

| 種類 | 場所 | Git 管理 |
|---|---|---|
| 提出用動画 | `<project>/videos/PAI最終課題_yoichiozaki.mp4` | する |
| 提出用レポート | `<project>/reports/説明資料_PAI最終課題_yoichiozaki.pdf` | する |
| 学習済みモデル | `<project>/checkpoints/<run>/final_model.zip`（+ `run_info.json`, `progress.csv`） | する（小さいもののみ） |
| 実験結果 | `<project>/results/`（CSV/JSON/図） | する |
| 中間ファイル | `*/results/tmp/`, `*/videos/frames/` など | しない |
