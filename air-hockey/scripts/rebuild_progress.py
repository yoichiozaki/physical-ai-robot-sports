"""Rebuild an SB3 progress.csv from the stdout training log (the tables SB3 prints).

Used when progress.csv is incomplete (e.g. the file was replaced while training was running).

    python scripts/rebuild_progress.py checkpoints/logs/ppo_hybrid_s0.log \
        checkpoints/ppo_hybrid_s0/progress.csv
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import pandas as pd

ROW = re.compile(r"^\|\s+(\S[^|]*?)\s+\|\s*([^|]*?)\s*\|$")


def parse(log: Path) -> pd.DataFrame:
    rows, block, section = [], {}, ""
    for line in log.read_text(encoding="utf-8", errors="replace").splitlines():
        if line.startswith("---"):
            if block:
                rows.append(block)
            block, section = {}, ""
            continue
        m = ROW.match(line)
        if not m:
            continue
        key, val = m.group(1).strip(), m.group(2).strip()
        if key.endswith("/") and not val:
            section = key
            continue
        try:
            block[section + key] = float(val)
        except ValueError:
            continue
    if block:
        rows.append(block)
    df = pd.DataFrame(rows)
    if "time/total_timesteps" in df:
        df = df.sort_values("time/total_timesteps", kind="stable").reset_index(drop=True)
    return df


if __name__ == "__main__":
    log, out = Path(sys.argv[1]), Path(sys.argv[2])
    df = parse(log)
    df.to_csv(out, index=False)
    print(f"{out}: {len(df)} rows, max steps {df['time/total_timesteps'].max():.0f}")
