"""Generate the VLA demonstration dataset (LeRobot v3 format) from the physics expert.

Runs inside an environment with ``lerobot`` installed (Colab / the local LeRobot venv).
Each worker process writes its own dataset shard; the shards are merged at the end. Only
demonstrations in which the expert's return lands in the instructed zone are kept.

    python scripts/vla/make_dataset.py --out /content/data --episodes 600 --workers 8

Features (50 Hz): observation.images.front / front_prev (camera image now and 40 ms earlier,
the latter gives the policy the ball's motion), observation.state (8 joint positions + 8
joint velocities), action (commanded joint positions + velocities at t + 20 ms, and the aimed
landing point x, y), task (the language instruction).
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

JOINTS = ["gantry_x", "gantry_y", "q1", "q2", "q3", "q4", "q5", "q6"]
FPS = 50


BALL = ["ball.x", "ball.y", "ball.z", "ball.vx", "ball.vy", "ball.vz"]


def features(size: int, ball_state: bool = False) -> dict:
    img = {"dtype": "video", "shape": (size, size, 3), "names": ["height", "width", "channels"]}
    names = (
        [f"{j}.pos" for j in JOINTS] + [f"{j}.vel" for j in JOINTS] + (BALL if ball_state else [])
    )
    return {
        "observation.images.front": img,
        "observation.images.front_prev": dict(img),
        "observation.state": {"dtype": "float32", "shape": (len(names),), "names": names},
        "action": {
            "dtype": "float32", "shape": (18,),
            "names": [f"{j}.pos" for j in JOINTS] + [f"{j}.vel" for j in JOINTS]
            + ["aim_x", "aim_y"],
        },
    }  # fmt: skip


def plan(worlds: list[str], episodes: int, seed0: int) -> list[tuple[str, int]]:
    """(world, seed) jobs, interleaved so every shard gets every world."""
    return [(w, seed0 + i) for i in range(episodes) for w in worlds]


def _worker(args: tuple) -> dict:
    shard, jobs, out, size, after_hit, with_ball = args
    os.environ.setdefault("MUJOCO_GL", "egl")
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    from tennis.vla.episode import ExpertController, run_episode, world_sim
    from tennis.vla.render import PolicyCamera
    from tennis.vla.tasks import sample_task
    from tennis.vla.worlds import WORLDS

    root = Path(out) / "shards" / f"shard_{shard:02d}"
    if root.exists():
        shutil.rmtree(root)
    ds = LeRobotDataset.create(
        repo_id=f"local/tennis_vla_shard{shard}", fps=FPS, features=features(size, with_ball),
        root=root,
        robot_type="tennis_gantry_arm", use_videos=True, image_writer_threads=4,
    )  # fmt: skip
    cam = PolicyCamera(size)
    experts: dict = {}
    stats = {"kept": 0, "rejected": 0, "frames": 0, "by_world": {}}
    t0 = time.time()
    for world, seed in jobs:
        if world not in experts:
            experts[world] = ExpertController(world, world_sim(world))
        task = sample_task(world, WORLDS[world].gravity, seed)
        res = run_episode(task, experts[world], render=cam, record=True,
                          record_after_hit=after_hit)  # fmt: skip
        w = stats["by_world"].setdefault(world, {"kept": 0, "rejected": 0})
        if not res.in_zone or not res.frames:
            stats["rejected"] += 1
            w["rejected"] += 1
            continue
        for f in res.frames:
            ds.add_frame({
                "observation.images.front": f["front"],
                "observation.images.front_prev": f["front_prev"],
                "observation.state": np.concatenate([f["state"], f["ball"]]) if with_ball
                else f["state"],
                "action": f["action"],
                "task": task.instruction,
            })  # fmt: skip
        ds.save_episode()
        stats["kept"] += 1
        stats["frames"] += len(res.frames)
        w["kept"] += 1
    ds.finalize()
    cam.close()
    stats["seconds"] = round(time.time() - t0, 1)
    stats["root"] = str(root)
    return stats


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="/content/data")
    p.add_argument("--worlds", nargs="+", default=["earth", "mars", "moon"])
    p.add_argument("--episodes", type=int, default=600, help="tasks per world (before filtering)")
    p.add_argument("--seed0", type=int, default=0)
    p.add_argument("--size", type=int, default=384)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--after-hit", type=float, default=0.5, help="recorded seconds after contact")
    p.add_argument("--ball-state", action="store_true",
                   help="ablation: exact ball position/velocity appended to the state")  # fmt: skip
    a = p.parse_args(argv)
    jobs = plan(a.worlds, a.episodes, a.seed0)
    shards = [jobs[i :: a.workers] for i in range(a.workers)]
    t0 = time.time()
    with ProcessPoolExecutor(a.workers) as ex:
        tasks = [(i, s, a.out, a.size, a.after_hit, a.ball_state) for i, s in enumerate(shards)]
        results = list(ex.map(_worker, tasks))
    print(f"[dataset] shards done in {time.time() - t0:.0f}s", flush=True)
    from lerobot.datasets.dataset_tools import merge_datasets
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    parts = [
        LeRobotDataset(f"local/tennis_vla_shard{i}", root=r["root"]) for i, r in enumerate(results)
    ]
    merged_root = Path(a.out) / "tennis_vla"
    if merged_root.exists():
        shutil.rmtree(merged_root)
    merged = merge_datasets(parts, output_repo_id="local/tennis_vla", output_dir=merged_root)
    summary = {
        "episodes": merged.meta.total_episodes, "frames": merged.meta.total_frames,
        "kept": sum(r["kept"] for r in results), "rejected": sum(r["rejected"] for r in results),
        "by_world": {w: {k: sum(r["by_world"].get(w, {}).get(k, 0) for r in results)
                         for k in ("kept", "rejected")} for w in a.worlds},
        "seconds": round(time.time() - t0, 1), "size": a.size, "worlds": a.worlds,
        "episodes_per_world_requested": a.episodes, "seed0": a.seed0,
    }  # fmt: skip
    (Path(a.out) / "dataset_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
