"""Render demo clips of VLA episodes (Colab GPU): tennis-TV camera + the policy's camera inset
+ Japanese HUD (planet, instruction, controller, latency, result).

    python scripts/vla/render_vla_clips.py --checkpoint <ckpt> --clips clips.json --out /content/clips

clips.json: [{"name": "earth_vla", "world": "earth", "seed": 100003, "mode": "vla",
              "latency": 0.06}, ...]   mode: expert | vla | hier
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

MODE_JA = {
    "expert": "物理エキスパート（ボールの位置・速度を直接知る教師）",
    "vla": "VLA 単独（カメラ画像 → 関節の動き）",
    "hier": "階層型（VLA が狙いを決め、物理層が打つ）",
}


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--clips", required=True)
    p.add_argument("--out", default="/content/clips")
    p.add_argument("--device", default="cuda")
    a = p.parse_args(argv)
    os.environ.setdefault("MUJOCO_GL", "egl")
    import mujoco
    from evaluate_vla import SmolVLAPolicyFn

    from tennis.viz.hud import font
    from tennis.viz.video import VideoWriter
    from tennis.vla.episode import (
        ChunkPolicyController,
        ExpertController,
        HierarchicalController,
        run_episode,
        world_sim,
    )
    from tennis.vla.render import PolicyCamera
    from tennis.vla.tasks import INSTRUCTION_JA, sample_task
    from tennis.vla.worlds import WORLDS

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    fn = SmolVLAPolicyFn(a.checkpoint, a.device)
    cam = PolicyCamera(384)
    W, H, FPS, SIM_HZ = 1280, 720, 30, 50  # real-time playback, resampled to 30 fps
    renderers: dict[int, mujoco.Renderer] = {}
    for spec in json.loads(Path(a.clips).read_text()):
        world = WORLDS[spec["world"]]
        sim = world_sim(spec["world"])
        task = sample_task(spec["world"], world.gravity, int(spec["seed"]))
        mode, latency = spec["mode"], float(spec.get("latency", 0.0))
        ready = ExpertController(spec["world"], sim).player.ready_q
        fn.reset()
        if mode == "expert":
            ctrl = ExpertController(spec["world"], sim)
        elif mode == "vla":
            ctrl = ChunkPolicyController(fn, ready, latency=latency)
        else:
            ctrl = HierarchicalController(spec["world"], sim, fn, latency=latency)
        if id(sim) not in renderers:
            renderers[id(sim)] = mujoco.Renderer(sim.model, H, W)
        rend = renderers[id(sim)]
        path = out / f"{spec['name']}.mp4"
        frames: list[np.ndarray] = []

        start = float(spec.get("start", 0.0))  # [s] skip the beginning (slow worlds)

        def on_step(sim, res, imgs, rend=rend, frames=frames, start=start):
            if sim.time < start:
                return
            rend.update_scene(sim.data, camera="behind_home")
            frames.append((rend.render().copy(), None if imgs is None else imgs["front"], res.hit))

        res = run_episode(task, ctrl, render=cam, on_step=on_step)
        verdict = ("◎ 指示どおりのゾーンに着地" if res.in_zone else
                   "○ 相手コートにイン（ゾーン外）" if res.in_court else
                   "× 打ち返せず" if not res.hit else "× アウト／ネット")  # fmt: skip
        lat_txt = "" if mode == "expert" else f"推論の遅れ {latency * 1000:.0f} ms"
        with VideoWriter(path, W, H, FPS) as vw:
            n_out = int(len(frames) * FPS / SIM_HZ) + int(0.8 * FPS)  # hold the end 0.8 s
            n = n_out
            for i in range(n_out):
                img, inset, _ = frames[min(round(i * SIM_HZ / FPS), len(frames) - 1)]
                im = Image.fromarray(img).convert("RGBA")
                ov = Image.new("RGBA", im.size, (0, 0, 0, 0))
                d = ImageDraw.Draw(ov)
                d.rounded_rectangle((16, 14, 820, 150), 10, fill=(12, 16, 26, 190))
                d.text((30, 22), f"{world.label_ja}（重力 {world.gravity:.2f} m/s²、地球の "
                       f"{world.time_scale:.2f} 倍の速さ）", font=font(24), fill=(245, 247, 250))  # fmt: skip
                d.text((30, 58), f"指示: “{task.instruction}”（{INSTRUCTION_JA[task.instruction]}）",
                       font=font(20), fill=(150, 200, 255))  # fmt: skip
                d.text((30, 90), MODE_JA[mode], font=font(20), fill=(255, 215, 120))
                if lat_txt:
                    d.text((30, 118), lat_txt, font=font(18), fill=(200, 205, 215))
                if inset is not None:
                    ins = Image.fromarray(inset).resize((270, 270))
                    im.paste(ins, (W - 290, H - 300))
                    d.rectangle(
                        (W - 291, H - 301, W - 19, H - 29), outline=(255, 255, 255, 220), width=2
                    )
                    d.text(
                        (W - 290, H - 328), "VLA の入力画像", font=font(18), fill=(255, 255, 255)
                    )
                if i > n - 40:
                    d.rounded_rectangle((W // 2 - 260, H - 90, W // 2 + 260, H - 30), 12,
                                        fill=(0, 0, 0, 200))  # fmt: skip
                    d.text(
                        (W // 2, H - 60), verdict, font=font(26), fill=(255, 255, 255), anchor="mm"
                    )
                vw.write(np.asarray(Image.alpha_composite(im, ov).convert("RGB")))
        meta = {**spec, **res.summary()}
        path.with_suffix(".json").write_text(json.dumps(meta, ensure_ascii=False, indent=2))
        print(json.dumps(meta, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
