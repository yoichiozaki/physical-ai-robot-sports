"""Smoke test for local and Google Colab runtimes.

Checks: Python/library versions, GPU visibility, MuJoCo simulation + headless rendering,
a tiny PPO training run, checkpoint save/reload. Prints a JSON report (and writes it to
``--out``). Runs in about a minute on CPU.

    python scripts/colab_smoke_test.py --out results/colab/smoke_local.json
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import tempfile
import time
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--out", default=None)
    p.add_argument("--steps", type=int, default=2048)
    a = p.parse_args(argv)
    report: dict = {"python": platform.python_version(), "platform": platform.platform()}
    t0 = time.time()
    import torch

    report["torch"] = torch.__version__
    report["cuda_available"] = torch.cuda.is_available()
    report["gpu"] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    report["colab"] = "COLAB_RELEASE_TAG" in os.environ or Path("/content").exists()

    from hockey.infrastructure.doctor import check_render
    from hockey.simulation import HockeySim

    sim = HockeySim()
    t = time.time()
    for _ in range(250):  # 5 s of game time
        sim.step()
    report["sim_realtime_factor"] = round(5.0 / (time.time() - t), 1)
    try:
        report["render"] = check_render()
    except Exception as exc:  # report, do not hide: rendering is optional for training
        report["render"] = f"FAILED: {exc!r}"

    from hockey.learning.train import load_train_config, train

    with tempfile.TemporaryDirectory() as tmp:
        cfg = load_train_config(
            Path(__file__).resolve().parents[1] / "configs/training/smoke.yaml",
            {"total_timesteps": a.steps, "n_envs": 2},
        )
        t = time.time()
        run_dir = train(cfg, Path(tmp))
        report["train_seconds"] = round(time.time() - t, 1)
        from stable_baselines3 import PPO

        model = PPO.load(str(run_dir / "final_model.zip"), device="cpu")
        report["reload_ok"] = model.num_timesteps >= a.steps
    report["total_seconds"] = round(time.time() - t0, 1)
    report["ok"] = bool(report["reload_ok"]) and not str(report["render"]).startswith("FAILED")
    text = json.dumps(report, indent=2)
    print(text)
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(text, encoding="utf-8")
    return 0 if report["reload_ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
