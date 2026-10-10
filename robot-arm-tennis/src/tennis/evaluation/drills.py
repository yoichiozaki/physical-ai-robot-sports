"""Ball-machine drills: a feeder launches balls to the home robot, which plays a shot.

Used for impact-model identification, residual-learning data collection and accuracy tests.
Every hit is recorded with the plan (intended shot), the measured contact, the measured
outgoing velocity/spin and the landing point.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from tennis.control.ballistics import G
from tennis.control.planner import ShotSpec
from tennis.control.players import PhysicsPlayer
from tennis.evaluation.conditions import NOMINAL, ConditionConfig
from tennis.evaluation.match import make_view
from tennis.learning.residual import plan_features
from tennis.sim.game import GameConfig, Referee
from tennis.sim.kinematics import world_to_player
from tennis.sim.physics import TennisSim

PARKED = np.array([0.0, 1.9, 0.0, 1.2, -2.4, 0.0, 0.0, 0.0])  # away robot out of the way


@dataclass
class HitRecord:
    target: tuple[float, float]  # aim actually planned (after any learned correction)
    speed: float
    plan_t: float
    contact_t: float
    ball_pos: np.ndarray  # planned contact ball position (player frame)
    v_in: np.ndarray  # measured incoming velocity at contact (player frame)
    normal: np.ndarray
    racket_speed: float
    v_out_plan: np.ndarray
    v_out: np.ndarray  # measured outgoing velocity (corrected back to the end of contact)
    spin: np.ndarray
    landing: np.ndarray | None  # first bounce (player frame) or None (net / no bounce)
    net: bool
    face_offset: np.ndarray | None = None  # contact point on the face (handle axis, across) [m]
    q_contact: np.ndarray | None = None  # planned joint configuration at contact
    features: np.ndarray | None = None  # residual-model features of the executed plan
    intended: tuple[float, float] | None = None  # target chosen by the strategy (before correction)

    @property
    def clean(self) -> bool:
        """Contact well inside the string bed (the impact model does not cover frame hits)."""
        return (
            self.face_offset is not None
            and abs(self.face_offset[0]) < 0.09
            and abs(self.face_offset[1]) < 0.075
        )


def random_feed(rng: np.random.Generator, gravity: float = G) -> tuple[np.ndarray, np.ndarray]:
    """A ball-machine shot from the far baseline into the home court (world = home frame).

    With another ``gravity`` the flight time scales by sqrt(G / gravity) and the speeds by its
    inverse, so the (drag-free) path is geometrically the same - only slower or faster."""
    p0 = np.array([2.7, rng.uniform(-0.7, 0.7), rng.uniform(0.7, 1.1)])
    tgt = np.array([rng.uniform(-2.7, -1.4), rng.uniform(-0.9, 0.9)])
    T = rng.uniform(0.8, 1.2) * float(np.sqrt(G / gravity))
    v0 = np.array(
        [(tgt[0] - p0[0]) / T, (tgt[1] - p0[1]) / T, (0.033 - p0[2]) / T + 0.5 * gravity * T]
    )
    return p0, v0


def random_shot(rng: np.random.Generator) -> ShotSpec:
    return ShotSpec((rng.uniform(1.2, 2.7), rng.uniform(-0.9, 0.9)), rng.uniform(3.8, 7.0), "drill")


def run_drill(
    sim: TennisSim,
    player: PhysicsPlayer,
    n: int,
    seed: int = 0,
    condition: ConditionConfig = NOMINAL,
    shot_fn=random_shot,
    resample: bool = True,
) -> list[HitRecord]:
    """``resample=False`` keeps one sampled condition for the whole drill (a "session" on one
    court), which is what online adaptation needs."""
    rng = np.random.default_rng(seed)
    records: list[HitRecord] = []
    root_x = sim.cfg.gantry.root_x
    params, npos, nvel = condition.sample(rng)
    for k in range(n):
        if k > 0 and resample:
            params, npos, nvel = condition.sample(rng)
        sim.set_params(params)
        player.reset(np.random.default_rng(rng.integers(2**31)), keep_adaptation=k > 0)
        shot = shot_fn(rng)
        player.strategy = lambda v, p, r, s=shot: s
        sim.reset(robot_q={"home": player.ready_q, "away": PARKED})
        sim.set_trajectory("home", player.traj)
        ref = Referee(sim, GameConfig(), rng)
        ref.live, ref.last_hitter, ref.server, ref.ball_id = True, "away", "away", 1
        p0, v0 = random_feed(rng, sim.cfg.physics.gravity)
        sim.place_ball(p0, v0)
        plan = None
        hit = None
        face = None
        out = None
        landing = None
        net = False
        for _ in range(250):
            st = sim.state()
            view = make_view(st, "home", ref, rng, (npos, nvel), root_x)
            traj, _ = player.act(view)
            if traj is not None:
                sim.set_trajectory("home", traj)
            if player.plan is not None:
                plan = player.plan
            res = sim.step()
            for e in res.events:
                if e.kind == "racket_home" and hit is None:
                    hit = (e.time, world_to_player("home", e.ball_vel))
                    sid = sim._site["home"]
                    rot = sim.data.site_xmat[sid].reshape(3, 3)
                    face = (rot.T @ (e.ball_pos - sim.data.site_xpos[sid]))[:2]
                elif e.kind == "floor" and hit is not None and landing is None:
                    landing = world_to_player("home", e.ball_pos)
                elif e.kind == "net" and hit is not None:
                    net = True
                elif e.kind.startswith("body"):
                    res.lost = True
            if hit is not None and out is None and not sim._touch["racket_home"]:
                s = sim.state()
                dt = s.time - (hit[0] + 0.005)
                v = world_to_player("home", s.ball_vel)
                player.observe_outgoing(v + rng.normal(0.0, nvel, 3) if nvel > 0 else v, dt)
                g = sim.cfg.physics.gravity
                v_corr = v + dt * (np.array([0.0, 0.0, g]) + 0.0202 * np.linalg.norm(v) * v)
                out = (v_corr, world_to_player("home", s.ball_spin))
            if landing is not None or res.lost or (hit is not None and sim.time > hit[0] + 3.0):
                break
            if hit is None and plan is not None and sim.time > plan.t_contact + 0.5:
                break
        if plan is None or hit is None or out is None or plan.v_out is None:
            continue
        if landing is not None and landing[0] > 0:
            player.observe_landing(landing)
        records.append(
            HitRecord(
                target=plan.shot.target,
                speed=plan.shot.speed,
                plan_t=plan.t_contact,
                contact_t=hit[0],
                ball_pos=plan.ball_pos,
                v_in=hit[1],
                normal=plan.normal,
                racket_speed=plan.racket_speed,
                v_out_plan=plan.v_out,
                v_out=out[0],
                spin=out[1],
                landing=landing,
                net=net,
                face_offset=face,
                q_contact=plan.q_contact.copy(),
                features=plan_features(plan),
                intended=shot.target,
            )
        )
    return records
