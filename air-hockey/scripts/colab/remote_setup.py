"""Executed *inside* a Colab runtime via ``colab exec -s NAME -f scripts/colab/remote_setup.py``.

Self-contained (``colab exec -f`` ships only this file): extracts the uploaded source
bundle, installs pinned dependencies (keeping Colab's preinstalled CUDA PyTorch) and
runs the smoke test on the remote runtime.
"""

import subprocess
import sys
import tarfile
from pathlib import Path

BUNDLE = Path("/content/colab_bundle.tar.gz")
ROOT = Path("/content/physical-ai-hockey")


def sh(cmd: list[str]) -> None:
    print("+", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


if not ROOT.exists():
    with tarfile.open(BUNDLE) as tar:
        tar.extractall("/content", filter="data")
sh([sys.executable, "-m", "pip", "install", "-q", "-r", str(ROOT / "requirements/colab.txt")])
sh([sys.executable, "-m", "pip", "install", "-q", "--no-deps", "-e", str(ROOT)])
sh(["nvidia-smi"]) if Path("/usr/bin/nvidia-smi").exists() else print("no GPU visible")
sh(
    [
        sys.executable,
        str(ROOT / "scripts/colab_smoke_test.py"),
        "--out",
        "/content/results/smoke_colab.json",
    ]
)
