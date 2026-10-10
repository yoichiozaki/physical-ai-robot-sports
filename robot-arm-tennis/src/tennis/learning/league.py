"""Self-play league (AlphaStar-style, simplified) for the per-shot tennis environment.

The learner (warm-started from the Hybrid AI D) plays against frozen snapshots of itself.
Every ``snapshot_every`` decisions the current policy is added to the league. Opponents are
sampled per episode: a scripted opponent with probability ``p_scripted`` (keeps the agent
honest against the original benchmark), otherwise the latest snapshot with probability
``p_latest`` (≈ self-play) or a past snapshot by prioritized fictitious self-play (PFSP:
weight (1 - p_win)^power, i.e. opponents the learner still loses points to are preferred).

``league.json`` (in the league directory) is written atomically by the trainer callback and
read by the environment processes at every reset.
"""

from __future__ import annotations

import json
import os
import shutil
from pathlib import Path

import numpy as np

LEAGUE_FILE = "league.json"


def read_league(league_dir: str | Path) -> dict:
    path = Path(league_dir) / LEAGUE_FILE
    if not path.exists():
        return {"snapshots": []}
    for _ in range(5):  # tolerate a concurrent atomic replace
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
    return {"snapshots": []}


def write_league(league_dir: str | Path, league: dict) -> None:
    path = Path(league_dir) / LEAGUE_FILE
    tmp = path.with_suffix(f".tmp{os.getpid()}")
    tmp.write_text(json.dumps(league, indent=2), encoding="utf-8")
    os.replace(tmp, path)


def win_rate(s: dict, prior: float = 4.0) -> float:
    """Learner's point-win rate against snapshot ``s`` (shrunk towards 0.5)."""
    return (s.get("points_won", 0) + 0.5 * prior) / (s.get("points", 0) + prior)


def pfsp_weights(snapshots: list[dict], power: float = 2.0, floor: float = 0.05) -> np.ndarray:
    w = np.array([(1.0 - win_rate(s)) ** power + floor for s in snapshots], dtype=float)
    return w / w.sum()


def init_league(league_dir: str | Path, init_policy: str | Path, name: str = "gen_000") -> dict:
    d = Path(league_dir)
    d.mkdir(parents=True, exist_ok=True)
    dst = d / f"{name}.zip"
    shutil.copyfile(init_policy, dst)
    gen0 = {"name": name, "path": str(dst), "timesteps": 0}
    gen0.update(points_won=0, points=0, games=0)
    league = {"snapshots": [gen0], "scripted": {}}
    write_league(d, league)
    return league


def make_league_callback(league_dir: str | Path, snapshot_every: int):
    from stable_baselines3.common.callbacks import BaseCallback

    class LeagueCallback(BaseCallback):
        """Adds snapshots to the league and aggregates the learner's results per opponent."""

        def __init__(self) -> None:
            super().__init__()
            self.dir = Path(league_dir)
            self.next_snapshot = snapshot_every

        def _on_step(self) -> bool:
            league = None
            for done, info in zip(self.locals["dones"], self.locals["infos"], strict=True):
                if not done or "points" not in info:
                    continue
                league = league or read_league(self.dir)
                opp, won, pts = info["opponent"], int(info["points_won"]), int(info["points"])
                rec = next((s for s in league["snapshots"] if s["name"] == opp), None)
                if rec is None:
                    rec = league.setdefault("scripted", {}).setdefault(
                        opp, {"points_won": 0, "points": 0, "games": 0}
                    )
                rec["points_won"] += won
                rec["points"] += pts
                rec["games"] += 1
            if self.num_timesteps >= self.next_snapshot:
                league = league or read_league(self.dir)
                k = len(league["snapshots"])
                path = self.dir / f"gen_{k:03d}.zip"
                self.model.save(str(path))
                league["snapshots"].append(
                    {"name": f"gen_{k:03d}", "path": str(path),
                     "timesteps": int(self.num_timesteps),
                     "points_won": 0, "points": 0, "games": 0}
                )  # fmt: skip
                self.next_snapshot += snapshot_every
                print(f"[league] gen_{k:03d} at {self.num_timesteps} decisions", flush=True)
            if league is not None:
                write_league(self.dir, league)
                snaps = league["snapshots"]
                self.logger.record("league/size", len(snaps))
                self.logger.record("league/win_rate_vs_latest", win_rate(snaps[-1]))
                self.logger.record("league/win_rate_vs_gen0", win_rate(snaps[0]))
            return True

    return LeagueCallback()
