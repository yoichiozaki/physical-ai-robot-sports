"""Executed *inside* a Colab runtime via ``colab exec -s NAME -f scripts/colab/remote_train.py``.

Environment variables (``--env K=V``): RUN_CONFIG, RUN_NAME, DRIVE_DIR.
Checkpoints are written to Google Drive (``colab drivemount`` first) so that a dropped
runtime can be resumed: if checkpoints already exist for RUN_NAME, training resumes from
the newest one. Results are packed to /content/results/<RUN_NAME>.tar.gz for download.
"""

import os
import re
import subprocess
import sys
import tarfile
from pathlib import Path

ROOT = Path("/content/physical-ai-hockey")
config = os.environ.get("RUN_CONFIG", "configs/training/ppo_hybrid.yaml")
run_name = os.environ.get("RUN_NAME", "ppo_hybrid_colab")
drive = Path(os.environ.get("DRIVE_DIR", "/content/drive/MyDrive/physical-ai-hockey"))
out_root = drive / "checkpoints" if drive.parent.exists() else Path("/content/checkpoints")
run_dir = out_root / run_name
cmd = [sys.executable, str(ROOT / "scripts/train.py"), "--config", str(ROOT / config),
       "--out", str(out_root), "--run-name", run_name]  # fmt: skip
ckpts = sorted(
    (run_dir / "checkpoints").glob("ckpt_*_steps.zip"),
    key=lambda p: int(re.search(r"_(\d+)_steps", p.name).group(1)),
)
if ckpts:
    cmd += ["--resume", str(ckpts[-1])]
print("+", " ".join(cmd), flush=True)
subprocess.run(cmd, check=True, cwd=ROOT)
Path("/content/results").mkdir(exist_ok=True)
with tarfile.open(f"/content/results/{run_name}.tar.gz", "w:gz") as tar:
    for name in (
        "final_model.zip",
        "best_model.zip",
        "progress.csv",
        "run_info.json",
        "config.json",
        "eval",
    ):
        if (run_dir / name).exists():
            tar.add(run_dir / name, arcname=f"{run_name}/{name}")
print("packed", f"/content/results/{run_name}.tar.gz")
