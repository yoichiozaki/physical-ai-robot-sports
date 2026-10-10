"""Start a long job in the background on a Colab VM (run with ``colab exec -f``).

Environment (``colab exec --env``): JOB_CMD (shell command), JOB_LOG (log file),
JOB_CWD (working directory, default /content/repo/robot-arm-tennis). Prints the PID.
"""

import os
import subprocess

cmd = os.environ["JOB_CMD"]
log = os.environ.get("JOB_LOG", "/content/job.log")
cwd = os.environ.get("JOB_CWD", "/content/repo/robot-arm-tennis")
env = {**os.environ, "MUJOCO_GL": "egl", "PYTHONUNBUFFERED": "1", "TOKENIZERS_PARALLELISM": "false"}
with open(log, "ab") as fh:
    proc = subprocess.Popen(["bash", "-c", cmd], cwd=cwd, stdout=fh, stderr=subprocess.STDOUT,
                            env=env, start_new_session=True)  # fmt: skip
print(f"started pid {proc.pid}: {cmd} -> {log}")
