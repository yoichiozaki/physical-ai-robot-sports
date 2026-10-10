"""Episode runner for instructed return drills with pluggable controllers.

Controllers receive a :class:`Context` every control step (50 Hz) and may return a new joint
trajectory for the home robot:

* :class:`ExpertController`        physics expert (the teacher; B with the instructed aim)
* :class:`ReplayController`        open-loop replay of recorded expert actions (fidelity check)
* :class:`ChunkPolicyController`   end-to-end policy (VLA): observation -> chunk of joint targets,
                                   executed while the next chunk is being computed; inference
                                   latency is simulated in *simulation time* (the simulation
                                   is paused while the network runs, then the result is applied
                                   ``latency`` seconds after the observation it was computed from)
* :class:`HierarchicalController`  the policy only chooses the aim; the physics expert executes

Actions (per step k, time t_k = k dt): commanded joint position and velocity at t_k + dt, plus
the aimed landing point (x, y) -> 18 numbers.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np

from tennis.control.planner import ShotSpec
from tennis.control.players import PhysicsPlayer
from tennis.evaluation.drills import PARKED
from tennis.evaluation.match import make_view
from tennis.sim.game import GameConfig, Referee
from tennis.sim.kinematics import world_to_player
from tennis.sim.physics import PhysicsParams, TennisSim
from tennis.sim.trajectory import JointTrajectory
from tennis.vla.tasks import Task, in_court, in_zone
from tennis.vla.worlds import WORLDS, make_expert

ACTION_DIM = 18
STATE_DIM = 16


@dataclass
class Context:
    sim: TennisSim
    task: Task
    step: int
    time: float
    q: np.ndarray
    qd: np.ndarray
    view: object  # PlayerView of the home robot (privileged ball state; experts only)
    images: Callable[[], dict[str, np.ndarray]] | None = None  # lazily rendered camera images


@dataclass
class EpisodeResult:
    task: Task
    hit: bool = False
    landing: np.ndarray | None = None
    net: bool = False
    body: bool = False
    t_hit: float | None = None
    steps: int = 0
    frames: list[dict] = field(default_factory=list)  # recorded observations / actions
    infer_calls: int = 0
    infer_seconds: float = 0.0
    min_dist: float = float("inf")  # closest racket-centre to ball distance before any hit [m]
    t_min: float | None = None

    @property
    def in_court(self) -> bool:
        return in_court(self.landing)

    @property
    def in_zone(self) -> bool:
        return in_zone(self.landing, self.task.zone)

    def summary(self) -> dict:
        tgt = np.array(self.task.target)
        err = float(np.linalg.norm(self.landing[:2] - tgt)) if self.landing is not None else None
        return {
            "world": self.task.world, "seed": self.task.seed, "kind": self.task.kind,
            "instruction": self.task.instruction, "zone": "-".join(self.task.zone),
            "hit": int(self.hit), "in_court": int(self.in_court), "in_zone": int(self.in_zone),
            "net": int(self.net), "landing_error_m": err, "steps": self.steps,
            "infer_calls": self.infer_calls,
            "infer_ms": 1e3 * self.infer_seconds / max(self.infer_calls, 1),
            "min_racket_ball_m": self.min_dist, "t_min": self.t_min,
        }  # fmt: skip


# ----------------------------------------------------------------------------- controllers
class ExpertController:
    """Physics player aiming at the task's target (speed scaled with the world)."""

    def __init__(self, world: str, sim: TennisSim) -> None:
        self.world = WORLDS[world]
        self.player: PhysicsPlayer = make_expert(self.world, sim.cfg, strategy=self._strategy)
        self.target: tuple[float, float] = (2.0, 0.0)

    def _strategy(self, view, pred, rng) -> ShotSpec:
        return ShotSpec(self.target, 5.5 * self.world.time_scale, "instructed")

    def reset(self, task: Task, rng: np.random.Generator) -> np.ndarray:
        self.target = task.target
        self.player.reset(rng)
        return self.player.ready_q

    def act(self, ctx: Context) -> JointTrajectory | None:
        traj, _ = self.player.act(ctx.view)
        return traj


def chunk_trajectory(start: tuple[np.ndarray, np.ndarray, np.ndarray], t_start: float,
                     t_first: float, q: np.ndarray, qd: np.ndarray, dt: float,
                     lo: np.ndarray, hi: np.ndarray) -> JointTrajectory:  # fmt: skip
    """Quintic segments through knots (q[i], qd[i]) at t_first + i dt, from the commanded state
    ``start`` at ``t_start`` (knot accelerations from finite differences of qd)."""
    q = np.clip(q, lo, hi)
    acc = np.gradient(qd, dt, axis=0) if len(qd) > 1 else np.zeros_like(qd)
    traj = JointTrajectory.hold(start[0], t_start)
    t_prev = t_start
    for i in range(len(q)):
        t_i = t_first + i * dt
        if t_i <= t_prev + 1e-9:
            continue
        kw = {"start": start, "t_start": t_start} if t_prev == t_start else {}
        traj.append(q[i], qd[i], t_i - t_prev, "chunk", a1=acc[i], **kw)
        t_prev = t_i
    traj.append(q[-1], np.zeros_like(q[-1]), 0.3, "settle")  # come to rest after the chunk
    return traj


class ReplayController:
    """Replays a recorded action sequence open-loop (perfect imitation, zero latency)."""

    def __init__(self, actions: np.ndarray, ready_q: np.ndarray) -> None:
        self.actions = np.asarray(actions)
        self.ready_q = ready_q

    def reset(self, task: Task, rng: np.random.Generator) -> np.ndarray:
        return self.ready_q

    def act(self, ctx: Context) -> JointTrajectory | None:
        if ctx.step != 0:
            return None
        sim = ctx.sim
        dt = sim.cfg.physics.control_dt
        start = sim.traj["home"].sample(ctx.time)
        lo, hi = _limits(sim)
        a = self.actions
        return chunk_trajectory(start, ctx.time, ctx.time + dt, a[:, :8], a[:, 8:16], dt, lo, hi)


def _limits(sim: TennisSim) -> tuple[np.ndarray, np.ndarray]:
    m = sim.model
    adr = sim._qadr["home"]
    jids = [int(np.nonzero(m.jnt_qposadr == a)[0][0]) for a in adr]
    rng = m.jnt_range[jids]
    return rng[:, 0], rng[:, 1]


Policy = Callable[[dict], np.ndarray]  # observation dict -> action chunk (N x ACTION_DIM)


def ball_state(view) -> np.ndarray:
    """Exact ball position and velocity (player frame) - privileged input for the ablation."""
    return np.concatenate([view.ball_pos, view.ball_vel]).astype(np.float32)


def observation(ctx: Context) -> dict:
    state = np.concatenate([ctx.q, ctx.qd])
    return {"images": ctx.images(), "state": state, "ball": ball_state(ctx.view),
            "task": ctx.task.instruction, "time": ctx.time}  # fmt: skip


class ChunkPolicyController:
    """End-to-end chunked policy with simulated inference latency (async execution).

    A query uses the observation at time t_q; its chunk becomes active at t_q + latency, and
    the actions that are already in the past at that moment are skipped. The next query is
    issued when the previous result arrives, but not more often than every ``min_period``.
    """

    def __init__(self, policy: Policy, ready_q: np.ndarray, latency: float,
                 min_period: float = 0.1) -> None:  # fmt: skip
        self.policy = policy
        self.ready_q = ready_q
        self.latency = latency
        self.min_period = min_period
        self.pending: tuple[float, np.ndarray] | None = None  # (t_obs, chunk)
        self.next_query = 0.0
        self.calls = 0
        self.seconds = 0.0

    def reset(self, task: Task, rng: np.random.Generator) -> np.ndarray:
        self.pending, self.next_query, self.calls, self.seconds = None, 0.0, 0, 0.0
        return self.ready_q

    def _query(self, ctx: Context) -> None:
        import time

        obs = observation(ctx)
        t0 = time.perf_counter()
        chunk = np.asarray(self.policy(obs))
        self.seconds += time.perf_counter() - t0
        self.calls += 1
        self.pending = (ctx.time, chunk)
        self.next_query = ctx.time + max(self.latency, self.min_period) - 1e-9

    def act(self, ctx: Context) -> JointTrajectory | None:
        out = None
        if self.pending is not None and ctx.time >= self.pending[0] + self.latency - 1e-9:
            t_obs, chunk = self.pending
            self.pending = None
            sim = ctx.sim
            dt = sim.cfg.physics.control_dt
            start = sim.traj["home"].sample(ctx.time)
            lo, hi = _limits(sim)
            out = chunk_trajectory(start, ctx.time, t_obs + dt, chunk[:, :8], chunk[:, 8:16],
                                   dt, lo, hi)  # fmt: skip
        if self.pending is None and ctx.time >= self.next_query:
            self._query(ctx)
            if self.latency <= 0.0:  # synchronous: apply immediately
                return self.act(ctx)
        return out


class HierarchicalController(ExpertController):
    """The policy looks once (when the ball is fed) and chooses the aim; physics executes.

    Until the answer arrives (``latency`` later) the expert prepares with a default aim; if
    the answer comes before the swing is committed, the shot is re-planned with it."""

    def __init__(self, world: str, sim: TennisSim, policy: Policy, latency: float) -> None:
        super().__init__(world, sim)
        self.policy = policy
        self.latency = latency
        self.answer: tuple[float, tuple[float, float]] | None = None
        self.asked = False
        self.calls = 0
        self.seconds = 0.0

    def reset(self, task: Task, rng: np.random.Generator) -> np.ndarray:
        q = super().reset(task, rng)
        self.target = (2.0, 0.0)  # default until the policy has answered
        self.answer, self.asked, self.calls, self.seconds = None, False, 0, 0.0
        return q

    def act(self, ctx: Context) -> JointTrajectory | None:
        import time

        if not self.asked:
            self.asked = True
            obs = observation(ctx)
            t0 = time.perf_counter()
            chunk = np.asarray(self.policy(obs))
            self.seconds += time.perf_counter() - t0
            self.calls += 1
            self.answer = (ctx.time + self.latency, (float(chunk[0, 16]), float(chunk[0, 17])))
        if self.answer is not None and ctx.time >= self.answer[0] - 1e-9:
            self.target = self.answer[1]
            self.answer = None
            p = self.player
            committed = p.plan is not None and ctx.time >= p.plan.t_swing_start - p.pcfg.commit_lead
            if not committed:
                p.shot = None  # re-choose the shot with the new aim
                p.last_plan_time = -1.0
        return super().act(ctx)


# ----------------------------------------------------------------------------- runner
_SIMS: dict[str, TennisSim] = {}


def world_sim(world: str) -> TennisSim:
    from tennis.vla.worlds import sim_config

    if world not in _SIMS:
        _SIMS[world] = TennisSim(sim_config(WORLDS[world]))
    return _SIMS[world]


def run_episode(task: Task, controller, render: Callable[[TennisSim], np.ndarray] | None = None,
                record: bool = False, prev_lag: int = 2,
                record_after_hit: float | None = None,
                on_step: Callable[[TennisSim, EpisodeResult, dict | None], None] | None = None,
                ) -> EpisodeResult:  # fmt: skip
    """Play one instructed return. ``render(sim)`` returns the camera image (needed by vision
    policies and for recording); ``record`` stores observations and expert actions (until
    ``record_after_hit`` seconds after the racket contact - Earth time, scaled with the world)."""
    sim = world_sim(task.world)
    world = WORLDS[task.world]
    dt = sim.cfg.physics.control_dt
    rng = np.random.default_rng(task.seed + 7)
    sim.set_params(PhysicsParams())
    ready = controller.reset(task, rng)
    away_q = PARKED.copy()
    from tennis.vla.tasks import opponent_gantry_y

    away_q[0] = 1.0  # parked a little in front of its baseline: easier to see from our side
    away_q[1] = opponent_gantry_y(task.opp_y)
    sim.reset(robot_q={"home": ready, "away": away_q})
    sim.place_ball(np.array(task.feed_p0), np.array(task.feed_v0))
    ref = Referee(sim, GameConfig(), rng)
    ref.live, ref.last_hitter, ref.server, ref.ball_id = True, "away", "away", 1
    res = EpisodeResult(task)
    history: list[np.ndarray] = []
    max_steps = int(4.6 / world.time_scale / dt)
    t_end = None
    root_x = sim.cfg.gantry.root_x
    for k in range(max_steps):
        st = sim.state()
        recording = record and not (
            record_after_hit is not None and res.hit
            and st.time > res.t_hit + record_after_hit / world.time_scale
        )  # fmt: skip
        imgs: dict[str, np.ndarray] | None = None
        if render is not None and (recording or not record):
            now = render(sim)  # every step: keeps the "previous frame" history continuous
            history.append(now)
            imgs = {"front": now, "front_prev": history[max(len(history) - 1 - prev_lag, 0)]}
        view = make_view(st, "home", ref, rng, (0.0, 0.0), root_x)
        ctx = Context(sim, task, k, st.time, st.q["home"], st.qd["home"], view,
                      (lambda imgs=imgs: imgs) if imgs is not None else None)  # fmt: skip
        traj = controller.act(ctx)
        if traj is not None:
            sim.set_trajectory("home", traj)
        if recording:
            q_d, qd_d, _ = sim.traj["home"].sample(st.time + dt)
            res.frames.append({
                "front": imgs["front"] if imgs else None,
                "front_prev": imgs["front_prev"] if imgs else None,
                "state": np.concatenate([st.q["home"], st.qd["home"]]).astype(np.float32),
                "ball": ball_state(view),
                "action": np.concatenate([q_d, qd_d, task.target]).astype(np.float32),
                "t": st.time,
            })  # fmt: skip
        step = sim.step()
        res.steps = k + 1
        if not res.hit:
            d = float(np.linalg.norm(sim.data.site_xpos[sim._site["home"]] - sim.ball()[0]))
            if d < res.min_dist:
                res.min_dist, res.t_min = d, sim.time
        if on_step is not None:
            on_step(sim, res, imgs)
        for e in step.events:
            if e.kind == "racket_home" and not res.hit:
                res.hit, res.t_hit = True, e.time
                ref.last_hitter = "home"
            elif e.kind == "floor" and res.hit and res.landing is None:
                res.landing = world_to_player("home", e.ball_pos)
                t_end = sim.time + 0.1
            elif e.kind == "floor" and not res.hit and e.ball_pos[0] < 0.0:
                # the feed bounced in our court; a second bounce there ends the episode
                ref.bounces["home"] = ref.bounces.get("home", 0) + 1
                if ref.bounces["home"] >= 2:
                    t_end = sim.time
            elif e.kind == "net" and res.hit:
                res.net = True
                t_end = sim.time + 0.1
            elif e.kind.startswith("body") and res.hit and res.landing is None:
                res.body = True
                t_end = sim.time + 0.1
        if step.lost:
            break
        if res.hit and res.landing is None and t_end is None:
            t_end = res.t_hit + 2.0 / world.time_scale
        if t_end is not None and sim.time >= t_end:
            break
    if hasattr(controller, "calls"):
        res.infer_calls, res.infer_seconds = controller.calls, controller.seconds
    return res
