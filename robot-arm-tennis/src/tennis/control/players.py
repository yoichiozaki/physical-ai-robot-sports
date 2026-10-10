"""Players: physics-based (B, B+, opponents), naive reactive baseline (A), strategy hooks."""

from __future__ import annotations

import time
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, field, replace

import numpy as np

from tennis.control.ballistics import BallModel, BallPredictor, CourtGeometry, Prediction
from tennis.control.planner import (
    HitPlan,
    PlannerConfig,
    RacketCommand,
    ShotSpec,
    SwingPlanner,
    contact_candidates,
)
from tennis.sim.config import TennisConfig
from tennis.sim.trajectory import JointTrajectory


@dataclass
class PlayerView:
    """What one robot knows (player frame; own joints in the root frame)."""

    time: float
    ball_pos: np.ndarray
    ball_vel: np.ndarray
    q: np.ndarray
    qd: np.ndarray
    opp_racket: np.ndarray
    opp_base: np.ndarray  # opponent gantry carriage (x, y) in this player's frame
    last_hitter: str  # "me" | "opp" | "none"
    my_toss: bool  # my serve: the toss is in the air / has bounced once
    point_live: bool
    bounced_my_side: bool
    score: tuple[int, int] = (0, 0)  # (me, opponent)
    ball_id: int = 0  # increments with every hit / new point


@dataclass
class DecisionInfo:
    mode: str = "ready"
    strategy: str = ""
    predicted_path: list[tuple[float, float, float]] = field(default_factory=list)
    bounce_points: list[tuple[float, float, float]] = field(default_factory=list)
    contact: tuple[float, float, float] | None = None
    contact_time: float | None = None  # seconds from now
    target: tuple[float, float] | None = None
    shot_path: list[tuple[float, float, float]] = field(default_factory=list)
    extra: dict[str, float] = field(default_factory=dict)


Strategy = Callable[[PlayerView, Prediction, np.random.Generator], "ShotSpec | RacketCommand"]


def centre_strategy(speed: float = 5.5, depth: float = 2.0) -> Strategy:
    return lambda view, pred, rng: ShotSpec((depth, 0.0), speed, "centre")


def rule_strategy(speed: float = 6.0) -> Strategy:
    """Aim at the corner farthest from the opponent's racket (open-court heuristic)."""

    def choose(view: PlayerView, pred: Prediction, rng: np.random.Generator) -> ShotSpec:
        oy = float(view.opp_racket[1])
        side = -1.0 if oy > 0 else 1.0
        depth = 2.5 if float(view.opp_racket[0]) < 2.6 else 1.6
        return ShotSpec((depth, 0.8 * side), speed, "open-court")

    return choose


def random_strategy(speed: tuple[float, float] = (4.0, 6.5)) -> Strategy:
    def choose(view: PlayerView, pred: Prediction, rng: np.random.Generator) -> ShotSpec:
        return ShotSpec(
            (rng.uniform(1.2, 2.6), rng.uniform(-0.9, 0.9)), rng.uniform(*speed), "random"
        )

    return choose


@dataclass(frozen=True)
class PlayerConfig:
    name: str = "Physics"
    replan_period: float = 0.06
    horizon: float = 2.2
    commit_lead: float = 0.03  # stop re-planning this long before the swing starts
    let_out_margin: float = 0.06  # leave balls predicted to land out by more than this
    delay: float = 0.0  # perception latency [s]
    aim_noise: float = 0.0  # target noise [m] (scripted opponents)
    speed_scale: float = 1.0


class PhysicsPlayer:
    """Predict the incoming ball, pick a reachable contact, solve the shot, swing (B)."""

    def __init__(
        self,
        cfg: TennisConfig,
        player_cfg: PlayerConfig | None = None,
        strategy: Strategy | None = None,
        model: BallModel | None = None,
        planner_cfg: PlannerConfig | None = None,
    ) -> None:
        self.cfg = cfg
        self.pcfg = player_cfg or PlayerConfig()
        self.name = self.pcfg.name
        court = CourtGeometry(
            cfg.court.length / 2, cfg.court.width / 2, cfg.court.net_height, cfg.court.net_half_span
        )
        self.court = court
        self.predictor = BallPredictor(model or BallModel(), court, dt=0.005)
        self.shot_predictor = BallPredictor(model or BallModel(), court, dt=0.01)
        self.planner = SwingPlanner(cfg, planner_cfg, self.pcfg.speed_scale)
        self.strategy = strategy or centre_strategy()
        self.ready_q = self._ready_pose()
        self.aim_correction: Callable[[HitPlan | None, ShotSpec], ShotSpec] | None = None
        self.impact_adapter = None  # tennis.learning.adaptation.ImpactAdapter (optional)
        self.model_adapter = None  # tennis.learning.adaptation.ModelAdapter (optional)
        self.service_line: float | None = None  # set by the match (GameConfig.service_line)
        self.rng = np.random.default_rng(0)
        self.reset(self.rng)

    def _ready_pose(self) -> np.ndarray:
        kin = self.planner.kin
        gx, gy = 0.35, 0.45
        p = kin.shoulder(gx, gy) + np.array([0.52, -0.42, -0.22])
        n = np.array([1.0, 0.0, 0.15])
        q = kin.ik(p, n / np.linalg.norm(n), (gx, gy))
        assert q is not None, "ready pose must be reachable"
        return q

    def reset(self, rng: np.random.Generator, keep_adaptation: bool = False) -> None:
        self.rng = rng
        self.traj = JointTrajectory.hold(self.ready_q.copy())
        self.plan: HitPlan | None = None
        self.ball_id = -1
        self.shot: ShotSpec | RacketCommand | None = None
        self.t_commit: float | None = None
        self.hit_done = False
        self.mode = "ready"
        self.last_plan_time = -1.0
        self.recovering = False
        self._buf: deque[PlayerView] = deque(
            maxlen=max(1, round(self.pcfg.delay / self.cfg.physics.control_dt)) + 1
        )
        self.last_info = DecisionInfo()
        self.last_latency = 0.0
        self.last_pred: Prediction | None = None
        self._goal: tuple[float, float] | None = None
        self.last_hit_plan: HitPlan | None = None
        if (
            self.aim_correction is not None
            and hasattr(self.aim_correction, "reset")
            and not keep_adaptation
        ):
            self.aim_correction.reset()
        if self.impact_adapter is not None and not keep_adaptation:
            self.impact_adapter.reset(self.planner)
        if self.model_adapter is not None:
            self.model_adapter.reset(self, keep=keep_adaptation)

    def observe_outgoing(self, v_measured: np.ndarray, dt_since_contact: float) -> None:
        """Feedback: measured velocity of our last shot shortly after it left the racket."""
        if self.impact_adapter is not None and self.last_hit_plan is not None:
            self.impact_adapter.observe(
                self.last_hit_plan,
                v_measured,
                dt_since_contact,
                self.predictor.model.k_drag,
                self.planner,
                self.predictor.model.gravity,
            )

    def observe_landing(self, landing: np.ndarray) -> None:
        """Feedback: where our last shot bounced (player frame)."""
        if self.last_hit_plan is not None and hasattr(self.aim_correction, "observe"):
            self.aim_correction.observe(self.last_hit_plan, landing)

    def set_speed_scale(self, s: float) -> None:
        self.planner.set_speed_scale(s)

    def set_model(self, model: BallModel) -> None:
        self.predictor.set_model(model)
        self.shot_predictor.set_model(model)

    # ------------------------------------------------------------------ helpers
    def _delayed(self, view: PlayerView) -> PlayerView:
        """Perception latency compensated by the player's own (imperfect) internal model."""
        if self.pcfg.delay <= 0:
            return view
        self._buf.append(view)
        old = self._buf[0]
        lag = view.time - old.time
        if lag <= 0:
            return view
        pred = self.predictor.predict(old.ball_pos, old.ball_vel, horizon=lag + 0.01, max_bounces=1)
        pos, vel = pred.state_at(lag)
        return replace(view, ball_pos=pos, ball_vel=vel)

    def _choose_shot(self, view: PlayerView, pred: Prediction) -> ShotSpec | RacketCommand:
        shot = self.strategy(view, pred, self.rng)
        if isinstance(shot, ShotSpec) and self.pcfg.aim_noise > 0:
            tx, ty = shot.target
            ty = float(np.clip(ty + self.rng.normal(0, self.pcfg.aim_noise), -1.05, 1.05))
            tx = float(np.clip(tx + self.rng.normal(0, self.pcfg.aim_noise), 0.8, 2.85))
            shot = replace(shot, target=(tx, ty))
        if view.my_toss and self.service_line is not None and isinstance(shot, ShotSpec):
            # serve into the service box: every player's execution layer knows the rule
            tx, ty = shot.target
            shot = replace(shot, target=(min(tx, self.service_line - 0.35), ty),
                           speed=min(shot.speed, 5.0))  # fmt: skip
        return shot

    def _leave_it(self, pred: Prediction, view: PlayerView) -> bool:
        """Let balls go that will clearly bounce out (first bounce on my side outside the court)."""
        if view.my_toss or view.bounced_my_side:
            return False
        b = pred.first("bounce")
        if b is None or b.pos[0] > 0:
            return False
        x, y, _ = b.pos
        m = self.pcfg.let_out_margin
        return abs(y) > self.court.half_width + m or x < -self.court.half_length - m

    def _info(self, pred: Prediction | None, mode: str) -> DecisionInfo:
        info = DecisionInfo(mode=mode)
        if pred is not None:
            info.predicted_path = [tuple(p) for p in pred.pos[::4]]
            info.bounce_points = [e.pos for e in pred.bounces()]
            t_fc = 0.3
            if pred.t[-1] >= t_fc:
                fp, _ = pred.state_at(t_fc)
                info.extra.update(
                    {
                        "fx": float(fp[0]),
                        "fy": float(fp[1]),
                        "fz": float(fp[2]),
                        "f_due": self._now + t_fc,
                    }
                )
        plan = self.plan
        if plan is not None:
            info.contact = tuple(plan.ball_pos)
            info.contact_time = plan.t_contact - self._now
            shot = plan.shot
            info.strategy = shot.label
            if isinstance(shot, ShotSpec):
                info.target = shot.target
            if plan.v_out is not None or isinstance(shot, RacketCommand):
                v_out = (
                    plan.v_out
                    if plan.v_out is not None
                    else self.planner.impact_forward(plan.ball_vel, plan.normal, plan.racket_speed)
                )
                sp = self.shot_predictor.predict(plan.ball_pos, v_out, horizon=2.0, max_bounces=1)
                info.shot_path = [tuple(p) for p in sp.pos[::3]]
            info.extra.update({"racket_speed": plan.racket_speed, "swing_time": plan.t_swing})
        return info

    # ------------------------------------------------------------------ positioning
    def _gantry_goal(
        self, now: float, start, goal_xy: tuple[float, float]
    ) -> JointTrajectory | None:
        lo, hi = self.planner.lo, self.planner.hi
        gx = float(np.clip(goal_xy[0], lo[0], hi[0]))
        gy = float(np.clip(goal_xy[1], lo[1], hi[1]))
        if (
            self._goal is not None
            and abs(self._goal[0] - gx) < 0.05
            and abs(self._goal[1] - gy) < 0.05
        ):
            return None
        self._goal = (gx, gy)
        q = self.ready_q.copy()
        q[0], q[1] = gx, gy
        self.traj = self.planner.move_to(now, start, q, label="position")
        return self.traj

    def _preposition(self, now: float, start, cands, pred: Prediction) -> JointTrajectory | None:
        """No feasible swing yet: move the gantry toward the most promising contact point."""
        if not cands:
            return self._dodge(now, start, pred, None)
        c = cands[0]
        pr = self.planner.kin.player_to_root(c.pos)
        return self._gantry_goal(
            now, start, (pr[0] - self.planner.pc.forward[0], pr[1] + self.planner.pc.right[0])
        )

    def _dodge(
        self, now: float, start, pred: Prediction, view: PlayerView | None
    ) -> JointTrajectory | None:
        """Step aside if the ball will pass through the robot's body (letting it go)."""
        base = self.planner.kin.root_to_player(np.array([start[0][0], start[0][1], 0.0]))
        near = (
            (np.abs(pred.pos[:, 0] - base[0]) < 0.35)
            & (np.abs(pred.pos[:, 1] - base[1]) < 0.45)
            & (pred.pos[:, 2] < 1.3)
        )
        if not np.any(near):
            return None
        y_ball = float(pred.pos[np.argmax(near), 1])
        away = 1.0 if base[1] >= y_ball else -1.0
        return self._gantry_goal(now, start, (start[0][0], start[0][1] + away * 0.8))

    # ------------------------------------------------------------------ main
    def act(self, view: PlayerView) -> tuple[JointTrajectory | None, DecisionInfo]:
        t0 = time.perf_counter()
        if self.model_adapter is not None:
            self.model_adapter.observe(view, self)
        out = self._act(view)
        self.last_latency = time.perf_counter() - t0
        return out

    def _act(self, view: PlayerView) -> tuple[JointTrajectory | None, DecisionInfo]:
        self._now = now = view.time
        view = self._delayed(view)
        start = self.traj.sample(now)
        plan = self.plan
        # committed swing / follow-through
        if plan is not None and now >= plan.t_swing_start - self.pcfg.commit_lead:
            self.last_hit_plan = plan
            if now <= plan.t_contact + plan.t_swing:
                self.mode = "swing" if now <= plan.t_contact else "follow"
                return None, self._info(None, self.mode)
            self.plan = None
            self.hit_done = True
        incoming = view.point_live and (view.last_hitter == "opp" or view.my_toss)
        if view.ball_id != self.ball_id:
            self.ball_id = view.ball_id
            self.shot, self.t_commit, self.hit_done, self.plan = None, None, False, None
            self.last_plan_time = -1.0
            self._goal = None
        if not incoming or self.hit_done:
            if self.mode not in ("recover", "ready"):
                self.mode = "recover"
                self._goal = None
                self.traj = self.planner.move_to(now, start, self.ready_q)
                return self.traj, self._info(None, self.mode)
            if self.mode == "recover" and now > self.traj.t_end:
                self.mode = "ready"
            return None, self._info(None, self.mode)
        if now - self.last_plan_time < self.pcfg.replan_period:
            return None, self.last_info
        self.last_plan_time = now
        pred = self.predictor.predict(
            view.ball_pos, view.ball_vel, self.pcfg.horizon, max_bounces=2
        )
        self.last_pred = pred
        if self._leave_it(pred, view):
            self.mode = "let"
            self.plan = None
            info = self._info(pred, "let")
            self.last_info = info
            return self._dodge(now, start, pred, view), info
        if self.shot is None:
            self.shot = self._choose_shot(view, pred)
        shot = self.shot
        if self.aim_correction is not None and isinstance(shot, ShotSpec):
            shot = self.aim_correction(self.plan, shot)
        new_plan = None
        cands: list = []
        if self.t_commit is not None and self.t_commit - now > self.planner.pc.min_time * 0.5:
            t_rel = self.t_commit - now
            bp, bv = pred.state_at(t_rel)
            new_plan = self.planner.plan(now, start, t_rel, bp, bv, shot, self.shot_predictor)
        if new_plan is None:
            cands = contact_candidates(
                pred, self.planner.pc, self.planner.kin, self.planner.lo, self.planner.hi,
                after_bounce_only=view.my_toss, bounced_already=view.bounced_my_side,
            )  # fmt: skip
            for c in cands[: self.planner.pc.candidates]:
                new_plan = self.planner.plan(
                    now, start, c.t, c.pos, c.vel, shot, self.shot_predictor
                )
                if new_plan is not None:
                    self.t_commit = now + c.t
                    break
        if new_plan is None and self.plan is not None and self.plan.t_contact > now + 0.02:
            # The refined prediction gave no better feasible plan: keep the swing under way.
            info = self._info(pred, "prepare")
            self.last_info = info
            return None, info
        if new_plan is None:
            self.t_commit = None
            self.mode = "chase"
            self.plan = None
            moved = self._preposition(now, start, cands, pred)
            info = self._info(pred, "chase")
            self.last_info = info
            return moved, info
        self.plan = new_plan
        self.traj = new_plan.trajectory
        self.mode = "prepare"
        info = self._info(pred, "prepare")
        self.last_info = info
        return self.traj, info


class NaivePlayer(PhysicsPlayer):
    """Baseline A: no ball physics for returns. Assumes the opponent's ball keeps its current
    velocity in a straight line (no gravity, drag or bounce) and plays a fixed racket stroke.
    Its own serve is pre-programmed (the toss is known exactly), so it is not penalised there."""

    def __init__(self, cfg: TennisConfig, player_cfg: PlayerConfig | None = None) -> None:
        super().__init__(cfg, player_cfg or PlayerConfig(name="Naive"), strategy=None)
        self.fixed = RacketCommand(pitch=0.30, yaw=0.0, speed=4.0, label="fixed stroke")
        self.physics_predictor = self.predictor
        self.naive_predictor = _StraightLinePredictor(self.court)
        self.strategy = lambda v, p, r: (
            self.fixed if not v.my_toss else ShotSpec((2.0, 0.0), 5.0, "serve")
        )

    def _act(self, view: PlayerView) -> tuple[JointTrajectory | None, DecisionInfo]:
        self.predictor = self.physics_predictor if view.my_toss else self.naive_predictor
        return super()._act(view)

    def _leave_it(self, pred: Prediction, view: PlayerView) -> bool:
        return False


class _StraightLinePredictor(BallPredictor):
    def predict(
        self, pos, vel, horizon=2.0, spin=None, max_bounces=2, stop_at_x=None
    ) -> Prediction:
        dt = self.dt
        t = np.arange(0.0, horizon, dt)
        p = np.asarray(pos, float)[None, :] + t[:, None] * np.asarray(vel, float)[None, :]
        v = np.repeat(np.asarray(vel, float)[None, :], len(t), axis=0)
        return Prediction(dt, t, p, v, [], np.zeros(len(t), dtype=np.int8))


def opponent_factory(cfg: TennisConfig) -> dict[str, Callable[[], PhysicsPlayer]]:
    """Scripted opponents: physics players with human-like imperfections (reaction delay,
    aim noise, a slightly inaccurate ball model) and different tactics / robot speed."""
    model = BallModel(k_drag=0.018, bounce_e=0.75, bounce_mu=0.27)  # nearly (not exactly) right

    def corner(v: PlayerView, p: Prediction, r: np.random.Generator) -> ShotSpec:
        return ShotSpec(
            (2.45, 0.85 * (1 if r.random() < 0.5 else -1)), 6.5, "corner", net_margin=0.08
        )

    def lob(v: PlayerView, p: Prediction, r: np.random.Generator) -> ShotSpec:
        return ShotSpec((2.35, r.uniform(-0.4, 0.4)), 3.8, "lob")

    return {
        "steady": lambda: PhysicsPlayer(
            cfg,
            PlayerConfig("Steady", delay=0.10, aim_noise=0.10),
            centre_strategy(5.0, 2.1),
            model,
        ),
        "random": lambda: PhysicsPlayer(
            cfg, PlayerConfig("Random", delay=0.10), random_strategy(), model
        ),
        "aggressive": lambda: PhysicsPlayer(
            cfg, PlayerConfig("Aggressive", delay=0.08, aim_noise=0.08), corner, model
        ),
        "lob": lambda: PhysicsPlayer(
            cfg, PlayerConfig("Lob", delay=0.12, aim_noise=0.10), lob, model
        ),
        "slow": lambda: PhysicsPlayer(
            cfg, PlayerConfig("Slow", delay=0.15, speed_scale=0.8), centre_strategy(5.0), model
        ),
    }


OPPONENTS = ("steady", "random", "aggressive", "lob", "slow")
