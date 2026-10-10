# STATUS.md — リポジトリ全体の状態

最終更新: 2026-10-10

| プロジェクト | 状態 | 詳細 |
|---|---|---|
| [air-hockey](air-hockey/STATUS.md) | ✅ 完了 | 全 12 フェーズ完了。評価 2,800 試合、動画 148.0 s、PDF 6 ページ |
| [robot-arm-tennis](robot-arm-tennis/STATUS.md) | ✅ 完了 | 本編（評価 1,680 試合）＋発展編（自己対戦リーグ、惑星テニス × VLA）。動画 173.5 s、PDF 8 ページ |

成果物のファイル名はどちらも `videos/PAI最終課題_yoichiozaki.mp4` と
`reports/説明資料_PAI最終課題_yoichiozaki.pdf`（各プロジェクト内）。

## 検証状況（2026-10-10 に再実行）
- `air-hockey`: `python -m pytest -q` → 36 passed
- `robot-arm-tennis`: `python -m pytest -q` → 34 passed
- リポジトリ全体: `ruff check .` → All checks passed、`ruff format --check .` → 142 files already formatted
- 作業ツリーはクリーン（未コミットの変更なし）

## 共有インフラ
- Dev Container: `.devcontainer/`（コンテナ名 `pai-dev`、ワークスペース `/workspaces/physical-ai-robot-sports`）
- CI: `.github/workflows/ci.yml`（両プロジェクトの ruff + pytest）
- Google Colab: 公式 `google-colab-cli` 0.7.4、OAuth 認証済み（2026-10-09）。
  テニスの発展編（VLA）で A100 を使用し約 72 CU を消費（データ生成・SmolVLA 学習 2 回・評価）。
  残高 127.16 CU、稼働中のランタイムなし（すべて停止済み）。ホッケーは全てローカル CPU で実行（CU 0）。

## リポジトリ構成の変更（2026-10-09）
- ホッケーのプロジェクト一式をルートから `air-hockey/` に移動（`git mv`、履歴保持）。
- `robot-arm-tennis/` を追加。開発環境・依存関係・CI・Colab ツールは共有。
- ワークスペースのフォルダ名（`AdaptiveAirHockeyAI`）は VS Code で開いたままでは変更できないため据え置き。
  変更する場合は VS Code を閉じてフォルダ名を `physical-ai-robot-sports` などに変え、コンテナ `pai-dev` を作り直す。
