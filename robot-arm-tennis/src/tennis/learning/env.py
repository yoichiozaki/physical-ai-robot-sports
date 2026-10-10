"""Gymnasium environment: one decision per shot of the home robot.

The physics layer (prediction, contact selection, swing planning, servo control) runs inside
the environment; the agent is asked for a decision only when a new ball comes towards the
home robot (or it has to serve). This keeps the decision rate at ~1 Hz instead of 1 kHz and
lets PPO learn tactics instead of re-learning ballistics.

Action spaces:
* ``hybrid`` (D): MultiDiscrete [3, 3, 3] = lateral target, depth, shot speed -> ShotSpec,
  solved by the ballistic shot model (with the impact model and online identification).
* ``direct`` (C): Box(3) = racket-face pitch, yaw and racket speed -> RacketCommand, i.e. the
  agent must learn how the ball leaves the racket (no ballistic shot model).

Observation (player frame, roughly scaled to [-1, 1], 23 values): ball position and velocity,
the predicted first bounce, own gantry position, opponent base and racket, serve flag, and
the online-identified physics (deviation of restitution, friction, drag, wind and impact
gain from the nominal model; zeros without adaptation).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from tennis.control.ballistics import BallModel
from tennis.control.planner import RacketCommand, ShotSpec
from tennis.control.players import (
    OPPONENTS,
    PhysicsPlayer,
    PlayerConfig,
    PlayerView,
    opponent_factory,
)
from tennis.evaluation.conditions import CONDITIONS
from tennis.evaluation.match import Match
from tennis.sim.game import GameConfig
from tennis.sim.physics import TennisSim

HYBRID_Y = (-0.8, 0.0, 0.8)
HYBRID_X = (1.2, 2.0, 2.6)
HYBRID_V = (4.0, 5.5, 7.0)
PITCH = (-0.1, 0.7)
YAW = (-0.45, 0.45)
RACKET_SPEED = (2.5, 7.5)
OBS_DIM = 23
NOMINAL_MODEL = BallModel()


@dataclass(frozen=True)
class EnvConfig:
    mode: str = "hybrid"  # "hybrid" (D) | "direct" (C)
    opponents: tuple[str, ...] = OPPONENTS
    conditions: tuple[str, ...] = ("nominal", "randomized")
    points_per_episode: int = 4
    in_bonus: float = 0.1  # reward for each own shot landing in
    max_decisions: int = 80
    adaptive: bool = True  # online identification (+ gated residual) in the physics layer
    residual: str | None = "checkpoints/residual/model.pt"
    # self-play league (tennis.learning.league); None = scripted opponents only
    league_dir: str | None = None
    p_scripted: float = 0.2
    p_latest: float = 0.4
    pfsp_power: float = 2.0
    service_line: float | None = None  # tennis service-box rule (GameConfig.service_line)


def decode_action(mode: str, action) -> ShotSpec | RacketCommand:
    a = np.asarray(action)
    if mode == "hybrid":
        iy, ix, iv = (int(v) for v in a)
        return ShotSpec((HYBRID_X[ix], HYBRID_Y[iy]), HYBRID_V[iv], f"rl-{ix}{iy}{iv}")
    a = np.clip(a.astype(float), -1.0, 1.0)
    lerp = lambda r, u: r[0] + (u + 1.0) * 0.5 * (r[1] - r[0])  # noqa: E731
    return RacketCommand(lerp(PITCH, a[0]), lerp(YAW, a[1]), lerp(RACKET_SPEED, a[2]), "rl-direct")


def shot_obs(view: PlayerView, player: PhysicsPlayer) -> np.ndarray:
    pred = player.predictor.predict(view.ball_pos, view.ball_vel, 2.0, max_bounces=1)
    b = pred.first("bounce")
    bounce = (b.pos[0] / 3.0, b.pos[1] / 1.5, 1.0) if b is not None else (0.0, 0.0, 0.0)
    ad = np.zeros(6)
    ma, ia = getattr(player, "model_adapter", None), getattr(player, "impact_adapter", None)
    if ma is not None and ma._base is not None:
        m0 = ma._base
        ad[:5] = (
            (ma.e - m0.bounce_e) / 0.1,
            (ma.mu - m0.bounce_mu) / 0.2,
            (ma.k_drag - m0.k_drag) / m0.k_drag,
            ma.wind[0] / 2.0,
            ma.wind[1] / 2.0,
        )
    if ia is not None:
        ad[5] = (ia.k - 1.0) / 0.2
    p, v = view.ball_pos, view.ball_vel
    obs = np.concatenate(
        [
            [p[0] / 3.0, p[1] / 1.5, p[2] / 1.5],
            np.asarray(v) / 8.0,
            bounce,
            [view.q[0] / 1.5, view.q[1] / 1.5],
            [view.opp_base[0] / 4.0, view.opp_base[1] / 1.5],
            [view.opp_racket[0] / 4.0, view.opp_racket[1] / 1.5, view.opp_racket[2] / 1.5],
            [1.0 if view.my_toss else 0.0],
            np.clip(ad, -3.0, 3.0),
        ]
    )
    return np.clip(obs, -5.0, 5.0).astype(np.float32)


def needs_decision(view: PlayerView, decided_id: int) -> bool:
    return (
        view.point_live
        and (view.last_hitter == "opp" or view.my_toss)
        and view.ball_id != decided_id
    )


def make_agent_player(sim_cfg, adaptive: bool, residual: str | None, name: str) -> PhysicsPlayer:
    player = PhysicsPlayer(sim_cfg, PlayerConfig(name))
    if adaptive:
        from tennis.learning.adaptation import make_adaptive

        model = None
        if residual and Path(residual).exists():
            from tennis.learning.residual import ResidualModel

            model = ResidualModel.load(residual)
        make_adaptive(player, model)
    return player


class TennisEnv(gym.Env):
    metadata: ClassVar[dict] = {"render_modes": []}

    def __init__(self, cfg: EnvConfig | None = None) -> None:
        super().__init__()
        self.cfg = cfg = cfg or EnvConfig()
        self.sim = TennisSim()
        self.home = make_agent_player(self.sim.cfg, cfg.adaptive, cfg.residual, "Agent")
        self.home.strategy = self._strategy
        factory = opponent_factory(self.sim.cfg)
        self.opponents = {name: factory[name]() for name in cfg.opponents}
        if cfg.mode == "hybrid":
            self.action_space = spaces.MultiDiscrete([3, 3, 3])
        else:
            self.action_space = spaces.Box(-1.0, 1.0, (3,), np.float32)
        self.observation_space = spaces.Box(-5.0, 5.0, (OBS_DIM,), np.float32)
        self.match: Match | None = None
        self._snapshots: dict[str, PhysicsPlayer] = {}
        self._action = None
        self._decided_id = -1
        self._shots_seen = 0
        self._points_seen = 0
        self._decisions = 0
        self.episode_info: dict = {}

    # ------------------------------------------------------------------ strategy hook
    def _strategy(self, view, pred, rng):
        if self._action is None:
            return ShotSpec((2.0, 0.0), 5.5, "fallback")
        return decode_action(self.cfg.mode, self._action)

    # ------------------------------------------------------------------ gym API
    def _snapshot_player(self, snap: dict) -> PhysicsPlayer:
        player = self._snapshots.get(snap["name"])
        if player is None:
            from tennis.learning.policy import PolicyStrategy

            cfg = self.cfg
            player = make_agent_player(self.sim.cfg, cfg.adaptive, cfg.residual, snap["name"])
            player.strategy = PolicyStrategy(snap["path"], self.cfg.mode, player)
            self._snapshots[snap["name"]] = player
        return player

    def _pick_opponent(self, rng: np.random.Generator) -> tuple[str, PhysicsPlayer]:
        cfg = self.cfg
        if cfg.league_dir is not None and rng.random() >= cfg.p_scripted:
            from tennis.learning.league import pfsp_weights, read_league

            snaps = read_league(cfg.league_dir)["snapshots"]
            if snaps:
                if rng.random() < cfg.p_latest:
                    snap = snaps[-1]
                else:
                    snap = snaps[int(rng.choice(len(snaps), p=pfsp_weights(snaps, cfg.pfsp_power)))]
                return snap["name"], self._snapshot_player(snap)
        name = str(rng.choice(list(cfg.opponents)))
        return name, self.opponents[name]

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        rng = self.np_random
        opp, opp_player = self._pick_opponent(rng)
        cond = str(rng.choice(list(self.cfg.conditions)))
        if options:
            if "opponent" in options:
                opp = options["opponent"]
                opp_player = self.opponents[opp]
            cond = options.get("condition", cond)
        self.match = Match(
            self.sim, self.home, opp_player, seed=int(rng.integers(2**31)),
            condition=CONDITIONS[cond], first_server=str(rng.choice(["home", "away"])),
            stop_after_points=self.cfg.points_per_episode,
            game=GameConfig(service_line=self.cfg.service_line),
        )  # fmt: skip
        self._action = None
        self._decided_id = -1
        self._shots_seen = self._points_seen = self._decisions = 0
        self.episode_info = {"opponent": opp, "condition": cond}
        obs, _, terminated, _ = self._run_until_decision()
        if terminated:  # (practically impossible) no decision in the whole episode
            obs = np.zeros(OBS_DIM, np.float32)
        return obs, dict(self.episode_info)

    def step(self, action):
        self._action = action
        self._decisions += 1
        obs, reward, terminated, info = self._run_until_decision()
        truncated = not terminated and self._decisions >= self.cfg.max_decisions
        if truncated:
            pts = self.match.ref.points
            won = sum(p.winner == "home" for p in pts)
            info.update(points_won=won, points=len(pts), is_success=won > len(pts) - won)
        return obs, reward, terminated, truncated, info

    # ------------------------------------------------------------------ internals
    def _rewards(self) -> float:
        ref = self.match.ref
        r = 0.0
        for s in ref.shot_log:
            if s["player"] == "home" and s["landing"] in ("in", "volleyed") and not s.get("_rw"):
                s["_rw"] = True
                r += self.cfg.in_bonus
        while self._points_seen < len(ref.points):
            r += 1.0 if ref.points[self._points_seen].winner == "home" else -1.0
            self._points_seen += 1
        return r

    def _run_until_decision(self):
        m = self.match
        reward = 0.0
        for _ in range(200_000):
            if m.done:
                reward += self._rewards()
                won = sum(p.winner == "home" for p in m.ref.points)
                info = dict(self.episode_info, points_won=won, points=len(m.ref.points))
                info["is_success"] = won > len(m.ref.points) - won
                return np.zeros(OBS_DIM, np.float32), reward, True, info
            views = m.observe()
            v = views["home"]
            if needs_decision(v, self._decided_id):
                self._decided_id = v.ball_id
                reward += self._rewards()
                return shot_obs(v, self.home), reward, False, dict(self.episode_info)
            m.advance()
        raise RuntimeError("no decision point reached")
