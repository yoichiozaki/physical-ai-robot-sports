"""Pack the code needed on a Colab VM into dist/bundle.tar.gz (no checkpoints / results).

python scripts/vla/colab_bundle.py
"""

import tarfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]  # robot-arm-tennis/
INCLUDE = ["pyproject.toml", "README.md", "src", "scripts/vla", "configs"]


def keep(info: tarfile.TarInfo) -> tarfile.TarInfo | None:
    name = info.name
    if "__pycache__" in name or name.endswith((".pyc", ".egg-info")) or ".egg-info/" in name:
        return None
    return info


def main() -> None:
    out = ROOT / "dist" / "bundle.tar.gz"
    out.parent.mkdir(exist_ok=True)
    with tarfile.open(out, "w:gz") as tf:
        for item in INCLUDE:
            tf.add(ROOT / item, arcname=f"robot-arm-tennis/{item}", filter=keep)
    print(out, f"{out.stat().st_size / 1024:.0f} KB")


if __name__ == "__main__":
    main()
