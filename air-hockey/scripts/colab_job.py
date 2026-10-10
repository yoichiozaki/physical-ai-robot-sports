"""Plan / pack / run a Google Colab training job with approval and budget gates.

    python scripts/colab_job.py pack                       # source bundle -> dist/
    python scripts/colab_job.py plan --hours 1 --gpu T4    # print the exact commands + CU estimate
    python scripts/colab_job.py run  --hours 0.2 --i-approve-auth --i-approve-cu 1.0

``run`` executes the plan with the official Colab CLI. It refuses to start unless the
user explicitly approves the OAuth login and a CU budget, and it never retries failed
cloud steps (AGENTS.md section 4). CU usage is printed before/after for EXPERIMENTS.md.
"""

from __future__ import annotations

import argparse
import shlex
import subprocess
import sys

from hockey.infrastructure.colab import APPROVAL_CU_LIMIT, ColabJob, pack, plan


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("action", choices=["pack", "plan", "run"])
    p.add_argument("--session", default="hockey")
    p.add_argument("--gpu", default="T4", help="T4|L4|G4|H100|A100 or 'none' for a CPU runtime")
    p.add_argument("--config", default="configs/training/ppo_hybrid.yaml")
    p.add_argument("--run-name", default="ppo_hybrid_colab")
    p.add_argument("--hours", type=float, default=1.0)
    p.add_argument("--cu-per-hour", type=float, default=1.8, help="measure with `colab usage`")
    p.add_argument(
        "--i-approve-auth", action="store_true", help="user approved the Colab OAuth login"
    )
    p.add_argument("--i-approve-cu", type=float, default=0.0, help="CU budget approved by the user")
    a = p.parse_args(argv)
    job = ColabJob(
        session=a.session,
        gpu=None if a.gpu.lower() == "none" else a.gpu,
        config=a.config,
        run_name=a.run_name,
        hours=a.hours,
        cu_per_hour=a.cu_per_hour,
    )
    if a.action == "pack":
        print(pack())
        return 0
    steps = plan(job)
    cu = job.estimated_cu
    print(f"# estimated compute units: {cu:.1f} CU ({job.hours} h x {job.cu_per_hour} CU/h)")
    if a.action == "plan":
        print("\n".join(steps))
        return 0
    if not a.i_approve_auth:
        print(
            "refusing: Colab OAuth login needs the user's approval (--i-approve-auth)",
            file=sys.stderr,
        )
        return 2
    if job.estimated_cu > a.i_approve_cu:
        print(
            f"refusing: estimated {job.estimated_cu:.1f} CU exceeds the approved budget "
            f"{a.i_approve_cu} CU (jobs over {APPROVAL_CU_LIMIT} CU always need explicit approval)",
            file=sys.stderr,
        )
        return 2
    pack()
    for line in steps:
        cmd = line.split("#")[0].strip()
        if not cmd:
            continue
        print("+", cmd, flush=True)
        rc = subprocess.run(shlex.split(cmd)).returncode
        if rc != 0:  # no automatic retries of cloud steps
            print(f"step failed (exit {rc}); stopping the runtime", file=sys.stderr)
            subprocess.run(["colab", "stop", "-s", job.session])
            return rc
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
