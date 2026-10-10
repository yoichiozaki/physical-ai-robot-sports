"""PPO training (Stable-Baselines3) for the per-shot tennis environment, with periodic
evaluation, checkpoints, resume and a run_info.json describing the run."""

from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from tennis import ENV_VERSION, __version__
from tennis.config import from_dict, to_dict
from tennis.learning.env import EnvConfig, TennisEnv


@dataclass(frozen=True)
class EvalConfig:
    freq: int = 20_000  # decisions between evaluations
    episodes: int = 30
    seed: int = 100_000


@dataclass(frozen=True)
class TrainConfig:
    run_name: str = "ppo_hybrid"
    seed: int = 0
    total_timesteps: int = 120_000  # decisions (one per shot)
    n_envs: int = 9
    torch_threads: int = 1
    device: str = "cpu"
    checkpoint_freq: int = 20_000
    ppo: dict[str, Any] = field(default_factory=dict)
    env: EnvConfig = field(default_factory=EnvConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)  # episodes = 0 disables evaluation
    init_from: str | None = None  # warm start: copy the policy weights of this checkpoint
    snapshot_every: int = 0  # self-play league: add the current policy every N decisions
    league_seed: str | None = None  # first league member (default: init_from); exploiter target


def load_train_config(path: str | Path, overrides: dict[str, Any] | None = None) -> TrainConfig:
    import yaml

    data = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    data.update(overrides or {})
    return from_dict(TrainConfig, data)


def make_env_fn(env_cfg: EnvConfig, rank: int, seed: int):
    def _init():
        import torch
        from stable_baselines3.common.monitor import Monitor

        torch.set_num_threads(1)
        keywords = ("is_success", "opponent", "points_won", "points")
        env = Monitor(TennisEnv(env_cfg), info_keywords=keywords)
        env.reset(seed=seed + rank)
        return env

    return _init


def _git_commit() -> str | None:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, check=True
        )
        dirty = subprocess.run(
            ["git", "status", "--porcelain"], capture_output=True, text=True, check=True
        ).stdout
        return out.stdout.strip() + ("-dirty" if dirty.strip() else "")
    except (OSError, subprocess.CalledProcessError):
        return None


def versions() -> dict[str, str]:
    import gymnasium
    import mujoco
    import stable_baselines3
    import torch

    return {
        "python": platform.python_version(),
        "tennis": __version__,
        "env_version": ENV_VERSION,
        "mujoco": mujoco.__version__,
        "gymnasium": gymnasium.__version__,
        "stable_baselines3": stable_baselines3.__version__,
        "torch": torch.__version__,
        "numpy": np.__version__,
    }


def train(cfg: TrainConfig, out_root: Path, resume: Path | None = None,
          extra_steps: int | None = None) -> Path:  # fmt: skip
    import torch
    from stable_baselines3 import PPO
    from stable_baselines3.common.callbacks import CallbackList, CheckpointCallback, EvalCallback
    from stable_baselines3.common.logger import configure
    from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

    torch.set_num_threads(cfg.torch_threads)
    run_dir = out_root / cfg.run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.json").write_text(json.dumps(to_dict(cfg), indent=2), encoding="utf-8")
    if cfg.env.league_dir is not None and resume is None:
        from tennis.learning.league import init_league

        seed_policy = cfg.league_seed or cfg.init_from
        if seed_policy is None:
            raise ValueError("a self-play league needs init_from or league_seed (generation 0)")
        init_league(cfg.env.league_dir, seed_policy)
    vec_cls = SubprocVecEnv if cfg.n_envs > 1 else DummyVecEnv
    env = vec_cls([make_env_fn(cfg.env, i, cfg.seed * 1000) for i in range(cfg.n_envs)])
    cbs: list = [
        CheckpointCallback(
            save_freq=max(cfg.checkpoint_freq // cfg.n_envs, 1),
            save_path=str(run_dir / "checkpoints"),
            name_prefix="ckpt",
        )
    ]
    eval_env = None
    if cfg.eval.episodes > 0:
        eval_env = DummyVecEnv([make_env_fn(cfg.env, 0, cfg.eval.seed)])
        cbs.append(
            EvalCallback(
                eval_env,
                n_eval_episodes=cfg.eval.episodes,
                eval_freq=max(cfg.eval.freq // cfg.n_envs, 1),
                deterministic=True,
                best_model_save_path=str(run_dir),
                log_path=str(run_dir / "eval"),
                verbose=1,
            )
        )
    if cfg.snapshot_every > 0 and cfg.env.league_dir is not None:
        from tennis.learning.league import make_league_callback

        cbs.append(make_league_callback(cfg.env.league_dir, cfg.snapshot_every))
    callbacks = CallbackList(cbs)
    if resume is not None:
        model = PPO.load(str(resume), env=env, device=cfg.device)
        remaining = (
            extra_steps
            if extra_steps is not None
            else max(cfg.total_timesteps - model.num_timesteps, 0)
        )
        reset = False
        print(
            f"[train] resumed from {resume} at {model.num_timesteps}; {remaining} more", flush=True
        )
    else:
        model = PPO("MlpPolicy", env, seed=cfg.seed, device=cfg.device, verbose=1, **cfg.ppo)
        if cfg.init_from is not None:
            src = PPO.load(cfg.init_from, device=cfg.device)
            model.policy.load_state_dict(src.policy.state_dict())
            print(f"[train] policy initialised from {cfg.init_from}", flush=True)
        remaining = cfg.total_timesteps
        reset = True
    model.set_logger(configure(str(run_dir), ["stdout", "csv"]))
    t0 = time.time()
    model.learn(total_timesteps=remaining, callback=callbacks, reset_num_timesteps=reset)
    duration = time.time() - t0
    model.save(str(run_dir / "final_model.zip"))
    env.close()
    if eval_env is not None:
        eval_env.close()

    ckpt_name = "best_model.zip" if eval_env is not None else "final_model.zip"
    info_path = run_dir / "run_info.json"
    history = (
        json.loads(info_path.read_text(encoding="utf-8")).get("sessions", [])
        if info_path.exists()
        else []
    )
    history.append(
        {
            "resumed_from": str(resume) if resume else None,
            "steps_this_session": int(remaining),
            "duration_s": round(duration, 1),
            "steps_per_s": round(remaining / max(duration, 1e-9), 1),
            "finished_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        }
    )
    info = {
        "run_name": cfg.run_name,
        "algorithm": "PPO",
        "policy": "MlpPolicy",
        "seed": cfg.seed,
        "total_timesteps": int(model.num_timesteps),
        "timestep_unit": "decision (one per shot of the home robot)",
        "training_duration_s": round(sum(s["duration_s"] for s in history), 1),
        "sessions": history,
        "n_envs": cfg.n_envs,
        "cpu_count": os.cpu_count(),
        "git_commit": _git_commit(),
        "versions": versions(),
        "checkpoint": str((run_dir / ckpt_name).as_posix()),
        "config": to_dict(cfg),
    }
    info_path.write_text(json.dumps(info, indent=2), encoding="utf-8")
    print(f"[train] done: {model.num_timesteps} decisions in {duration / 60:.1f} min", flush=True)
    return run_dir


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="Train a PPO agent for Robot-Arm Tennis")
    p.add_argument("--config", required=True, help="YAML under configs/training/")
    p.add_argument("--out", default="checkpoints", help="output root directory")
    p.add_argument("--resume", default=None, help="checkpoint .zip to resume from")
    p.add_argument("--extra-steps", type=int, default=None)
    p.add_argument("--total-timesteps", type=int, default=None)
    p.add_argument("--n-envs", type=int, default=None)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--run-name", default=None)
    a = p.parse_args(argv)
    overrides = {
        k: v
        for k, v in {
            "total_timesteps": a.total_timesteps,
            "n_envs": a.n_envs,
            "seed": a.seed,
            "run_name": a.run_name,
        }.items()
        if v is not None
    }
    cfg = load_train_config(a.config, overrides)
    train(cfg, Path(a.out), Path(a.resume) if a.resume else None, a.extra_steps)


if __name__ == "__main__":
    main()
