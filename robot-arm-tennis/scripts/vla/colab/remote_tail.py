"""Show the tail of a job log and the GPU status (run with ``colab exec -f``).

Environment: JOB_LOG (log file), TAIL (lines, default 15), GREP (optional substring filter).
"""

import os
import subprocess

log = os.environ.get("JOB_LOG", "/content/job.log")
n = int(os.environ.get("TAIL", "15"))
grep = os.environ.get("GREP")
try:
    with open(log, errors="replace") as fh:
        lines = fh.read().replace("\r", "\n").splitlines()
except FileNotFoundError:
    lines = [f"(no log {log})"]
if grep:
    lines = [ln for ln in lines if grep in ln]
print("\n".join(lines[-n:]))
r = subprocess.run(["nvidia-smi", "--query-gpu=utilization.gpu,memory.used,memory.total",
                    "--format=csv,noheader"], capture_output=True, text=True)  # fmt: skip
running = subprocess.run(
    ["bash", "-c", "pgrep -fa 'lerobot|make_dataset|evaluate_vla' | grep -v pgrep | wc -l"],
    capture_output=True, text=True,
).stdout.strip()  # fmt: skip
print("gpu:", r.stdout.strip(), "| job processes:", running)
