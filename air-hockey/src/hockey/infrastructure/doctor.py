"""Environment self-check: ``python -m hockey.infrastructure.doctor``.

Verifies the Dev Container: Python/library versions, MuJoCo simulation, headless
offscreen rendering (EGL, falling back to OSMesa), FFmpeg and the Japanese font.
Exit code is non-zero if a required component is missing.
"""

from __future__ import annotations

import importlib
import os
import platform
import shutil
import subprocess
import sys


def check_render() -> str:
    import mujoco

    xml = (
        '<mujoco><worldbody><light pos="0 0 2"/><geom type="plane" size="1 1 .1"/>'
        '<geom type="sphere" size=".1" pos="0 0 .1" rgba="1 0 0 1"/></worldbody></mujoco>'
    )
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    renderer = mujoco.Renderer(model, 64, 64)
    try:
        renderer.update_scene(data)
        img = renderer.render()
    finally:
        renderer.close()
    if img.std() < 1:
        raise RuntimeError("rendered image is blank")
    return f"ok ({os.environ.get('MUJOCO_GL', 'default')})"


def main() -> int:
    ok = True
    print(f"python      {platform.python_version()} ({sys.executable})")
    for mod in (
        "mujoco",
        "gymnasium",
        "stable_baselines3",
        "torch",
        "numpy",
        "scipy",
        "matplotlib",
        "reportlab",
    ):
        try:
            m = importlib.import_module(mod)
            print(f"{mod:<18}{getattr(m, '__version__', '?')}")
        except ImportError as exc:
            ok = False
            print(f"{mod:<18}MISSING ({exc})")
    try:
        import torch

        print(f"{'cuda':<18}{torch.cuda.is_available()}")
    except ImportError:
        pass
    try:
        from hockey.simulation import HockeySim

        sim = HockeySim()
        for _ in range(10):
            sim.step()
        print(f"{'hockey sim':<18}ok (t={sim.time:.2f}s)")
    except Exception as exc:
        ok = False
        print(f"{'hockey sim':<18}FAILED: {exc!r}")
    try:
        print(f"{'offscreen render':<18}{check_render()}")
    except Exception as exc:
        ok = False
        hint = "(try MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa)"
        print(f"{'offscreen render':<18}FAILED: {exc!r}  {hint}")
    ff = shutil.which("ffmpeg")
    if ff:
        ver = subprocess.run([ff, "-version"], capture_output=True, text=True).stdout.splitlines()[
            0
        ]
        print(f"{'ffmpeg':<18}{ver}")
    else:
        ok = False
        print(f"{'ffmpeg':<18}MISSING")
    try:
        from hockey.visualization.hud import font

        font(12)
        print(f"{'japanese font':<18}ok")
    except FileNotFoundError as exc:
        ok = False
        print(f"{'japanese font':<18}MISSING ({exc})")
    print("doctor:", "OK" if ok else "PROBLEMS FOUND")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
