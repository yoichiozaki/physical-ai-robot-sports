"""Remote setup on a Colab VM (run with ``colab exec -s <session> -f``).

Unpacks /content/bundle.tar.gz (made by ``scripts/vla/colab_bundle.py``) into
/content/repo, installs LeRobot (SmolVLA) + MuJoCo + this package, and checks GPU, EGL
rendering and the imports. Prints one JSON line with the environment.
"""

import json
import os
import subprocess
import sys
import tarfile

# MuJoCo EGL rendering on the GPU needs the NVIDIA EGL vendor file (as in the MuJoCo Colab
# tutorial); without it rendering silently falls back to the CPU (~100x slower).
ICD = "/usr/share/glvnd/egl_vendor.d/10_nvidia.json"
if not os.path.exists(ICD):
    os.makedirs(os.path.dirname(ICD), exist_ok=True)
    with open(ICD, "w") as f:
        f.write('{"file_format_version": "1.0.0", "ICD": {"library_path": "libEGL_nvidia.so.0"}}\n')
os.makedirs("/content/repo", exist_ok=True)
with tarfile.open("/content/bundle.tar.gz") as tf:
    tf.extractall("/content/repo")
PKGS = ["lerobot[smolvla]==0.4.4", "mujoco==3.15.0", "scipy", "pyyaml", "pillow", "gymnasium"]
r = subprocess.run([sys.executable, "-m", "pip", "install", "-q", *PKGS], capture_output=True,
                   text=True)  # fmt: skip
print(r.stdout[-2000:], r.stderr[-3000:])
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "--no-deps", "-e",
                "/content/repo/robot-arm-tennis"], check=True)  # fmt: skip
# Japanese font for the HUD of the demo clips (scripts/vla/render_vla_clips.py)
FONT = "apt-get -qq update && apt-get -qq install -y fonts-ipaexfont-gothic"
subprocess.run(["bash", "-c", FONT], capture_output=True)
check = r"""
import os, json, time
os.environ["MUJOCO_GL"] = "egl"
import torch, mujoco, lerobot
from tennis.vla.episode import world_sim
from tennis.vla.render import PolicyCamera
sim = world_sim("earth")
cam = PolicyCamera(384)
t0 = time.time()
for _ in range(50):
    img = cam(sim)
render_ms = (time.time() - t0) / 50 * 1e3
print(json.dumps({"torch": torch.__version__, "cuda": torch.cuda.is_available(),
    "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    "mujoco": mujoco.__version__, "lerobot": lerobot.__version__, "render_ms": round(render_ms, 2),
    "img": list(img.shape), "cpus": os.cpu_count()}))
"""
r = subprocess.run([sys.executable, "-c", check], capture_output=True, text=True,
                   env={**os.environ, "MUJOCO_GL": "egl"})  # fmt: skip
print(r.stdout[-3000:], r.stderr[-3000:])
print(json.dumps({"setup": "done"}))
