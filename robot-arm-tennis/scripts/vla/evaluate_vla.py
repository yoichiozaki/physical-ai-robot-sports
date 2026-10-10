"""Closed-loop evaluation of a fine-tuned VLA (SmolVLA) on instructed return drills.

Runs where ``lerobot`` is installed (Colab GPU for the main run). For every world and
condition the same evaluation tasks (seeds disjoint from the training demonstrations) are
played by:

    expert        physics expert with privileged ball state (the teacher, reference)
    vla@L         VLA end-to-end (camera images + joint state + instruction -> joint targets),
                  inference latency L simulated in simulation time; L = 0 is "thinking time
                  zero", "real" uses the latency measured on this machine
    hier@L        VLA chooses the aim from images + instruction, the physics layer executes

    python scripts/vla/evaluate_vla.py --checkpoint <train dir>/checkpoints/last/pretrained_model \
        --episodes 100 --out /content/results
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from pathlib import Path

import numpy as np


class SmolVLAPolicyFn:
    """observation dict -> action chunk (chunk_size x 18), un-normalized."""

    def __init__(self, checkpoint: str, device: str = "cuda", ball_state: bool = False,
                 noise: str = "random") -> None:  # fmt: skip
        import torch
        from lerobot.policies.factory import make_pre_post_processors
        from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

        self.torch = torch
        self.device = device
        self.ball_state = ball_state
        self.noise_mode = noise  # random (new sample per query) | fixed (per episode) | zero
        self.noise = None
        self.episode = 0
        self.policy = SmolVLAPolicy.from_pretrained(checkpoint)
        self.policy.config.device = device
        self.policy.to(device).eval()
        self.pre, self.post = make_pre_post_processors(
            self.policy.config, pretrained_path=checkpoint,
            preprocessor_overrides={"device_processor": {"device": device}},
        )  # fmt: skip

    def reset(self) -> None:
        self.policy.reset()
        self.episode += 1
        if self.noise_mode in ("fixed", "zero"):
            cfg = self.policy.config
            shape = (1, cfg.chunk_size, cfg.max_action_dim)
            gen = self.torch.Generator().manual_seed(self.episode)
            noise = self.torch.randn(shape, generator=gen)
            self.noise = (noise * 0 if self.noise_mode == "zero" else noise).to(self.device)

    def __call__(self, obs: dict) -> np.ndarray:
        torch = self.torch

        def img(a: np.ndarray):
            return torch.from_numpy(np.ascontiguousarray(a)).permute(2, 0, 1).float()[None] / 255.0

        state = np.concatenate([obs["state"], obs["ball"]]) if self.ball_state else obs["state"]
        batch = {
            "observation.images.front": img(obs["images"]["front"]),
            "observation.images.front_prev": img(obs["images"]["front_prev"]),
            "observation.state": torch.from_numpy(state.astype(np.float32))[None],
            "task": [obs["task"]],
        }
        batch = self.pre(batch)
        with torch.inference_mode():
            actions = self.policy.predict_action_chunk(batch, noise=self.noise)
        actions = self.post(actions)
        if self.device == "cuda":
            torch.cuda.synchronize()
        return actions[0].float().cpu().numpy()


def measure_latency(fn: SmolVLAPolicyFn, obs: dict, n: int = 30) -> dict:
    for _ in range(5):
        fn(obs)
    ts = []
    for _ in range(n):
        t0 = time.perf_counter()
        fn(obs)
        ts.append(time.perf_counter() - t0)
    ts = np.array(ts)
    return {"mean_s": float(ts.mean()), "p95_s": float(np.percentile(ts, 95)), "n": n}


def run_condition(name: str, world: str, seeds: list[int], make_controller, cam) -> list[dict]:
    from tennis.vla.episode import run_episode
    from tennis.vla.tasks import sample_task
    from tennis.vla.worlds import WORLDS

    rows = []
    for seed in seeds:
        task = sample_task(world, WORLDS[world].gravity, seed)
        res = run_episode(task, make_controller(), render=cam)
        rows.append({**res.summary(), "condition": name})
    return rows


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--out", default="/content/results")
    p.add_argument("--worlds", nargs="+", default=["earth", "mars", "moon"])
    p.add_argument("--episodes", type=int, default=100)
    p.add_argument("--seed0", type=int, default=100_000)
    p.add_argument("--latencies", nargs="+", type=float, default=[0.0, 0.1, 0.2, 0.4])
    p.add_argument("--hier-latencies", nargs="+", type=float, default=[0.4])
    p.add_argument("--size", type=int, default=384)
    p.add_argument("--device", default="cuda")
    p.add_argument("--skip-expert", action="store_true")
    p.add_argument("--ball-state", action="store_true", help="model trained with --ball-state")
    p.add_argument("--min-period", type=float, default=0.1, help="shortest re-planning period [s]")
    p.add_argument("--noise", default="random", choices=["random", "fixed", "zero"],
                   help="flow-matching noise: per query / per episode / zero")  # fmt: skip
    p.add_argument("--no-real", action="store_true", help="do not add the measured latency")
    p.add_argument("--no-hier", action="store_true", help="skip the hierarchical controller")
    p.add_argument("--no-vla", action="store_true", help="skip the end-to-end controller")
    p.add_argument("--tag", default="", help="suffix for the condition names")
    p.add_argument("--real-latency", type=float, default=None,
                   help="use this measured latency [s] instead of measuring here")  # fmt: skip
    p.add_argument("--measure-only", action="store_true", help="measure the latency and exit")
    a = p.parse_args(argv)
    os.environ.setdefault("MUJOCO_GL", "egl")
    from tennis.vla.episode import (
        ChunkPolicyController,
        ExpertController,
        HierarchicalController,
        world_sim,
    )
    from tennis.vla.render import PolicyCamera
    from tennis.vla.tasks import sample_task
    from tennis.vla.worlds import WORLDS

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    cam = PolicyCamera(a.size)
    fn = SmolVLAPolicyFn(a.checkpoint, a.device, ball_state=a.ball_state, noise=a.noise)
    # measured latency on this machine (one query: preprocessing + 10 flow-matching steps)
    sim0 = world_sim("earth")
    task0 = sample_task("earth", WORLDS["earth"].gravity, a.seed0)
    sim0.reset()
    sim0.place_ball(np.array(task0.feed_p0), np.array(task0.feed_v0))
    im = cam(sim0)
    obs0 = {"images": {"front": im, "front_prev": im}, "state": np.zeros(16, np.float32),
            "ball": np.zeros(6, np.float32), "task": task0.instruction}  # fmt: skip
    if a.real_latency is None:
        lat = measure_latency(fn, obs0, n=100)
        import torch

        lat["device"] = torch.cuda.get_device_name(0) if a.device == "cuda" else "cpu"
    else:
        lat = {"mean_s": a.real_latency, "given": True}
    real = math.ceil(lat["mean_s"] / 0.02) * 0.02
    lat["simulated_real_latency_s"] = real
    (out / "latency.json").write_text(json.dumps(lat, indent=2))
    if a.measure_only:
        print(json.dumps(lat), flush=True)
        return
    print(f"[eval] latency {lat['mean_s'] * 1e3:.1f} ms -> simulated {real:.2f} s", flush=True)
    latencies = [] if a.no_vla else sorted(set(a.latencies) if a.no_real else {*a.latencies, real})
    hier_latencies = (
        []
        if a.no_hier
        else sorted(set(a.hier_latencies) if a.no_real else {*a.hier_latencies, real})
    )
    rows: list[dict] = []
    seeds = [a.seed0 + i for i in range(a.episodes)]
    for world in a.worlds:
        sim = world_sim(world)
        ready = ExpertController(world, sim).player.ready_q
        conds = []
        if not a.skip_expert:
            conds.append(("expert", lambda w=world, s=sim: ExpertController(w, s), False))
        for L in latencies:
            tag = f"vla@{L:.2f}" + ("(real)" if abs(L - real) < 1e-9 else "") + a.tag

            def make(L=L, ready=ready):
                fn.reset()
                return ChunkPolicyController(fn, ready, latency=L, min_period=a.min_period)

            conds.append((tag, make, True))
        for L in hier_latencies:
            tag = f"hier@{L:.2f}" + ("(real)" if abs(L - real) < 1e-9 else "") + a.tag

            def make_h(L=L, w=world, s=sim):
                fn.reset()
                return HierarchicalController(w, s, fn, latency=L)

            conds.append((tag, make_h, True))
        for name, make_ctrl, vision in conds:
            t0 = time.time()
            part = run_condition(name, world, seeds, make_ctrl, cam if vision else None)
            rows += part
            m = {k: np.mean([r[k] for r in part]) for k in ("hit", "in_court", "in_zone")}
            print(f"[eval] {world:5s} {name:18s} hit {m['hit']:.2f} in {m['in_court']:.2f} "
                  f"zone {m['in_zone']:.2f} ({time.time() - t0:.0f}s)", flush=True)  # fmt: skip
            (out / "episodes.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    print("[eval] done", flush=True)


if __name__ == "__main__":
    main()
