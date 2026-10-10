"""Google Colab as a remote compute backend (official ``google-colab-cli`` v0.7.4).

Verified commands (``colab --help`` inside the Dev Container, 2026-10-09):

    colab --auth oauth2 new -s NAME [--gpu T4|L4|G4|H100|A100] [--high-mem]
    colab upload -s NAME LOCAL REMOTE          (single files only)
    colab exec -s NAME -f FILE [--timeout S] [--env K=V]   (runs FILE's code in the kernel)
    colab drivemount -s NAME [PATH]
    colab download -s NAME REMOTE LOCAL
    colab stop -s NAME
    colab usage                                 (compute-unit balance and usage rate)

This module only *builds* bundles and command plans. Running them requires the user's
OAuth login and spends compute units, so ``scripts/colab_job.py run`` refuses to start
without explicit approval flags (see AGENTS.md section 4).
"""

from __future__ import annotations

import tarfile
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]  # the air-hockey project directory
REPO = ROOT.parent  # monorepo root (shared requirements/)
BUNDLE_ITEMS = ("src", "configs", "scripts", "pyproject.toml", "README.md")
APPROVAL_CU_LIMIT = 5.0  # jobs expected to exceed this need explicit approval


@dataclass(frozen=True)
class ColabJob:
    session: str = "hockey"
    gpu: str | None = "T4"  # None -> CPU runtime
    config: str = "configs/training/ppo_hybrid.yaml"
    run_name: str = "ppo_hybrid_colab"
    hours: float = 1.0
    cu_per_hour: float = 1.8  # placeholder: measure with `colab usage` (rates are not published)
    drive_dir: str = "/content/drive/MyDrive/physical-ai-hockey"

    @property
    def estimated_cu(self) -> float:
        return self.hours * self.cu_per_hour


def pack(out: Path = ROOT / "dist" / "colab_bundle.tar.gz") -> Path:
    """Create a source bundle for upload (no checkpoints/results/secrets)."""
    out.parent.mkdir(parents=True, exist_ok=True)

    def _filter(info: tarfile.TarInfo) -> tarfile.TarInfo | None:
        name = info.name
        if "__pycache__" in name or name.endswith((".pyc", ".log")) or "/.env" in name:
            return None
        return info

    with tarfile.open(out, "w:gz") as tar:
        for item in BUNDLE_ITEMS:
            path = ROOT / item
            if path.exists():
                tar.add(path, arcname=f"physical-ai-hockey/{item}", filter=_filter)
        # Shared pinned requirements live at the repository root.
        tar.add(REPO / "requirements", arcname="physical-ai-hockey/requirements", filter=_filter)
    return out


def plan(job: ColabJob) -> list[str]:
    """The exact command sequence (Training Workflow steps 1-9), with step comments."""
    gpu = f" --gpu {job.gpu}" if job.gpu else ""
    s = f"-s {job.session}"
    env = (
        f"--env RUN_CONFIG={job.config} --env RUN_NAME={job.run_name} "
        f"--env DRIVE_DIR={job.drive_dir}"
    )
    timeout = int(job.hours * 3600 + 600)
    bundle = "dist/colab_bundle.tar.gz"
    result = f"/content/results/{job.run_name}.tar.gz"
    steps = [
        (
            "python scripts/colab_smoke_test.py --out results/colab/smoke_local.json",
            "1 local smoke",
        ),
        ("python scripts/colab_job.py pack", "2 source bundle"),
        ("colab usage", "CU balance before"),
        (f"colab --auth oauth2 new {s}{gpu}", "runtime (OAuth once)"),
        (f"colab upload {s} {bundle} /content/colab_bundle.tar.gz", "2 sync code"),
        (f"colab exec {s} -f scripts/colab/remote_setup.py --timeout 1200", "3+4 deps + GPU smoke"),
        (f"colab drivemount {s}", "checkpoint persistence"),
        (f"colab exec {s} -f scripts/colab/remote_train.py --timeout {timeout} {env}", "5+6 train"),
        (f"colab download {s} {result} results/colab/{job.run_name}.tar.gz", "7+8 results"),
        (f"colab stop {s}", "9 shutdown"),
        ("colab usage", "CU balance after"),
    ]
    return [f"{cmd}  # {comment}" for cmd, comment in steps]
