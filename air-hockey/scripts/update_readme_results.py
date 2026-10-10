"""Insert the latest results table (results/summary.md) into README.md between the markers."""

from pathlib import Path

readme = Path("README.md")
text = readme.read_text(encoding="utf-8")
start, end = "<!-- RESULTS:START -->", "<!-- RESULTS:END -->"
table = Path("results/summary.md").read_text(encoding="utf-8").strip()
notes = Path("results/readme_findings.md")
body = "## 結果（要約）\n\n" + (
    notes.read_text(encoding="utf-8").strip() + "\n\n" if notes.exists() else ""
)
body += (
    "勝率 / 平均得失点差（60 秒の試合、相手 5 種 × 20 試合、評価シードは学習と非重複）:\n\n"
    + table
    + "\n\n"
)
body += "![win rate](results/figures/win_rate.png)\n"
body += "![generalization](results/figures/generalization.png)\n"
i, j = text.index(start) + len(start), text.index(end)
readme.write_text(text[:i] + "\n" + body + text[j:], encoding="utf-8")
print("README results updated")
