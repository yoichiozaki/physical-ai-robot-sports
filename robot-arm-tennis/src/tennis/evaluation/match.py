"""Run tennis matches between two players and collect metrics (home perspective)."""

from __future__ import annotations

import math
import time
from collections.abc import Callable
from dataclasses import dataclass, field, replace

import numpy as np

from tennis.control.players import DecisionInfo, PlayerView
from tennis.evaluation.conditions import NOMINAL, ConditionConfig
from tennis.sim.game import OTHER, PARK_POS, GameConfig, PointResult, Referee
from tennis.sim.kinematics import world_to_player
from tennis.sim.mjcf import PLAYERS
from tennis.sim.physics import StepResult, TennisSim, TennisState


@dataclass
class StepRecord:
    sim: TennisSim
    state: TennisState
    infos: dict[str, DecisionInfo]
    result: StepResult
    point: PointResult | None
    referee: Referee


def opponent_base(state: TennisState, me: str, root_x: float) -> np.ndarray:
    o = OTHER[me]
    gx, gy = state.q[o][0], state.q[o][1]
    base_world = world_to_player(o, np.array([gx - root_x, gy, 0.0]))  # o's frame -> world
    return world_to_player(me, base_world)[:2]


def make_view(state: TennisState, me: str, ref: Referee, rng: np.random.Generator,
              noise: tuple[float, float], root_x: float) -> PlayerView:  # fmt: skip
    pos = world_to_player(me, state.ball_pos)
    vel = world_to_player(me, state.ball_vel)
    if noise[0] > 0 or noise[1] > 0:
        pos = pos + rng.normal(0.0, noise[0], 3)
        vel = vel + rng.normal(0.0, noise[1], 3)
    lh = ref.last_hitter
    return PlayerView(
        time=state.time,
        ball_pos=pos,
        ball_vel=vel,
        q=state.q[me],
        qd=state.qd[me],
        opp_racket=world_to_player(me, state.racket_pos[OTHER[me]]),
        opp_base=opponent_base(state, me, root_x),
        last_hitter="none" if lh is None else ("me" if lh == me else "opp"),
        my_toss=ref.live and lh is None and ref.server == me,
        point_live=ref.live,
        bounced_my_side=ref.bounced_on(me),
        score=(ref.score[me], ref.score[OTHER[me]]),
        ball_id=ref.ball_id,
    )


@dataclass
class MatchResult:
    points_won: int = 0
    points_lost: int = 0
    rally_hits: list[int] = field(default_factory=list)
    reasons_won: dict[str, int] = field(default_factory=dict)
    reasons_lost: dict[str, int] = field(default_factory=dict)
    shots: int = 0
    shots_in: int = 0
    shots_out: int = 0
    shots_net: int = 0
    serves: int = 0
    serves_in: int = 0
    opportunities: int = 0  # opponent shots landing in our court or volleyable
    returns_in: int = 0
    landing_errors: list[float] = field(default_factory=list)
    prediction_errors: list[float] = field(default_factory=list)
    latencies_ms: list[float] = field(default_factory=list)
    strategy_counts: dict[str, int] = field(default_factory=dict)
    duration: float = 0.0

    def summary(self) -> dict:
        n = self.points_won + self.points_lost
        lat = np.array(self.latencies_ms) if self.latencies_ms else np.array([np.nan])
        nan = float("nan")
        return {
            "points_won": self.points_won,
            "points_lost": self.points_lost,
            "point_win_rate": self.points_won / n if n else nan,
            "win": int(self.points_won > self.points_lost),
            "draw": int(self.points_won == self.points_lost),
            "loss": int(self.points_won < self.points_lost),
            "avg_rally_hits": float(np.mean(self.rally_hits)) if self.rally_hits else nan,
            "shots": self.shots,
            "in_rate": self.shots_in / self.shots if self.shots else nan,
            "net_rate": self.shots_net / self.shots if self.shots else nan,
            "out_rate": self.shots_out / self.shots if self.shots else nan,
            "serve_in_rate": self.serves_in / self.serves if self.serves else nan,
            "opportunities": self.opportunities,
            "returns_in": self.returns_in,
            "return_rate": self.returns_in / self.opportunities if self.opportunities else nan,
            "landing_error_m": float(np.mean(self.landing_errors)) if self.landing_errors else nan,
            "prediction_error_m": float(np.mean(self.prediction_errors))
            if self.prediction_errors
            else nan,
            "latency_ms_mean": float(np.nanmean(lat)),
            "latency_ms_p95": float(np.nanpercentile(lat, 95)),
            "winners": self.reasons_won.get("not returned", 0),
            "unforced_errors": self.reasons_lost.get("out", 0)
            + self.reasons_lost.get("did not cross the net", 0),
            "timeouts": sum(
                v
                for d in (self.reasons_won, self.reasons_lost)
                for k, v in d.items()
                if "timeout" in k
            ),
            "duration_s": self.duration,
            **{f"strategy_{k}": v for k, v in sorted(self.strategy_counts.items())},
        }


class Tracker:
    """Home-side metrics from referee logs, decisions and ground-truth ball states."""

    def __init__(self, home_player) -> None:
        self.r = MatchResult()
        self.home = home_player
        self._pending: list[
            tuple[float, np.ndarray, int]
        ] = []  # (due time, forecast, contacts seen)
        self._contacts = 0
        self._last_due: float | None = None
        self._prev: tuple[float, np.ndarray] | None = None

    def update(self, rec: StepRecord) -> None:
        r, st = self.r, rec.state
        self._contacts += sum(1 for e in rec.result.events if e.kind != "net")
        info = rec.infos["home"]
        due = info.extra.get("f_due")
        if due is not None and due != self._last_due:
            self._last_due = due
            fc = np.array([info.extra["fx"], info.extra["fy"], info.extra["fz"]])
            self._pending.append((due, fc, self._contacts))
        keep = []
        cur_t, cur_p = st.time, world_to_player("home", st.ball_pos)
        for due, fc, n_c in self._pending:
            if cur_t + 1e-9 >= due:
                prev = self._prev
                if n_c == self._contacts and prev is not None and prev[0] <= due < cur_t + 1e-9:
                    f = (due - prev[0]) / max(cur_t - prev[0], 1e-9)
                    actual = prev[1] + f * (cur_p - prev[1])
                    r.prediction_errors.append(float(np.linalg.norm(fc - actual)))
            else:
                keep.append((due, fc, n_c))
        self._pending = keep
        self._prev = (cur_t, cur_p)
        if rec.point is not None:
            self._contacts += 1
            self._pending.clear()

    def finish(self, ref: Referee, duration: float) -> MatchResult:
        r = self.r
        r.duration = duration
        for p in ref.points:
            if p.winner == "home":
                r.points_won += 1
                r.reasons_won[p.reason] = r.reasons_won.get(p.reason, 0) + 1
            else:
                r.points_lost += 1
                r.reasons_lost[p.reason] = r.reasons_lost.get(p.reason, 0) + 1
            r.rally_hits.append(p.hits)
        log = ref.shot_log
        for i, s in enumerate(log):
            if s["player"] == "home":
                first_of_point = bool(s.get("serve"))
                r.shots += 1
                if first_of_point:
                    r.serves += 1
                if s["landing"] in ("in", "volleyed"):
                    r.shots_in += 1
                    r.serves_in += int(first_of_point)
                elif s["landing"] == "out":
                    r.shots_out += 1
                elif s["landing"] == "net/short":
                    r.shots_net += 1
                tgt = s.get("target")
                if tgt is not None and s["pos"] is not None:
                    r.landing_errors.append(math.dist(tgt, s["pos"]))
            elif s["landing"] in ("in", "volleyed"):
                r.opportunities += 1
                nxt = log[i + 1] if i + 1 < len(log) else None
                if (
                    nxt is not None
                    and nxt["player"] == "home"
                    and nxt["landing"] in ("in", "volleyed")
                ):
                    r.returns_in += 1
        return r


class Match:
    """A match driven one control step at a time (``observe`` then ``advance``), so that a
    Gymnasium environment can pause exactly when the home player has to choose a shot."""

    def __init__(
        self,
        sim: TennisSim,
        home,
        away,
        seed: int,
        condition: ConditionConfig = NOMINAL,
        game: GameConfig | None = None,
        on_step: Callable[[StepRecord], None] | None = None,
        first_server: str = "home",
        stop_after_points: int | None = None,
    ) -> None:
        rng = np.random.default_rng(seed)
        params, self.npos, self.nvel = condition.sample(rng)
        away_scale = params.speed_scale[1] * getattr(
            getattr(away, "pcfg", None), "speed_scale", 1.0
        )
        params = replace(params, speed_scale=(1.0, away_scale))
        self.params = params
        sim.set_params(params)
        self.sim = sim
        self.game = game or GameConfig()
        self.ref = Referee(sim, self.game, np.random.default_rng(rng.integers(2**31)))
        self.players = {"home": home, "away": away}
        for p in PLAYERS:
            self.players[p].reset(np.random.default_rng(rng.integers(2**31)))
            if hasattr(self.players[p], "service_line"):
                self.players[p].service_line = self.game.service_line
        if hasattr(away, "set_speed_scale"):
            away.set_speed_scale(away_scale)
        self.noise_rng = np.random.default_rng(rng.integers(2**31))
        sim.reset(robot_q={p: self.players[p].ready_q for p in PLAYERS})
        for p in PLAYERS:
            sim.set_trajectory(p, self.players[p].traj)
        self.ref.start_point(first_server)
        self.tracker = Tracker(home)
        self.target_points = stop_after_points or self.game.points
        self.root_x = sim.cfg.gantry.root_x
        self.on_step = on_step
        self.pending_out: dict[str, float] = {}
        self.state: TennisState | None = None
        self.views: dict[str, PlayerView] = {}
        self.next_toss: float | None = None

    @property
    def done(self) -> bool:
        return len(self.ref.points) >= self.target_points

    def observe(self) -> dict[str, PlayerView]:
        """Sense: the current state as seen (noisy) by each player."""
        self.state = state = self.sim.state()
        for p in PLAYERS:
            view = make_view(
                state, p, self.ref, self.noise_rng, (self.npos, self.nvel), self.root_x
            )
            t_hit = self.pending_out.get(p)
            if t_hit is not None and state.time >= t_hit + 0.03:  # observe our shot leaving
                del self.pending_out[p]
                if hasattr(self.players[p], "observe_outgoing"):
                    self.players[p].observe_outgoing(view.ball_vel, state.time - (t_hit + 0.006))
            self.views[p] = view
        return self.views

    def advance(self) -> StepRecord:
        """Decide (both players) and step the physics by one control period."""
        sim, ref, players, tracker = self.sim, self.ref, self.players, self.tracker
        infos = {}
        for p in PLAYERS:
            t0 = time.perf_counter()
            traj, info = players[p].act(self.views[p])
            if p == "home":
                tracker.r.latencies_ms.append((time.perf_counter() - t0) * 1e3)
                if info.strategy and info.mode == "prepare":
                    tracker.r.strategy_counts[info.strategy] = (
                        tracker.r.strategy_counts.get(info.strategy, 0) + 1
                    )
            if traj is not None:
                sim.set_trajectory(p, traj)
            infos[p] = info
        n_shots = len(ref.shot_log)
        res = sim.step()
        point = ref.update(res)
        for e in res.events:
            if e.kind.startswith("racket_"):
                self.pending_out[e.kind.split("_", 1)[1]] = e.time
            elif e.kind in ("floor", "net"):
                self.pending_out.clear()
        for s in ref.shot_log:  # feedback to the shooter: where its shot bounced
            if not s.get("observed") and s["landing"] in ("in", "out") and s["pos"] is not None:
                s["observed"] = True
                shooter = players[s["player"]]
                if hasattr(shooter, "observe_landing"):
                    shooter.observe_landing(
                        world_to_player(s["player"], np.array([*s["pos"], 0.0]))
                    )
        for s in ref.shot_log[n_shots:]:  # the shooter's intended target (accuracy metrics)
            plan = getattr(players[s["player"]], "plan", None)
            tgt = getattr(getattr(plan, "shot", None), "target", None) if plan is not None else None
            s["target"] = (
                None
                if tgt is None
                else tuple(world_to_player(s["player"], np.array([tgt[0], tgt[1], 0.0]))[:2])
            )
        rec = StepRecord(sim, self.state, infos, res, point, ref)
        tracker.update(rec)
        if self.on_step is not None:
            self.on_step(rec)
        if point is not None and not self.done:
            if self.game.point_gap > 0:  # park the ball; both robots recover before the toss
                sim.place_ball(PARK_POS, (0.0, 0.0, 0.0))
                self.next_toss = sim.time + self.game.point_gap
            else:
                ref.start_point()
        elif self.next_toss is not None and sim.time >= self.next_toss - 1e-9:
            self.next_toss = None
            ref.start_point()
        return rec

    def run(self) -> MatchResult:
        while not self.done:
            self.observe()
            self.advance()
        return self.finish()

    def finish(self) -> MatchResult:
        return self.tracker.finish(self.ref, self.sim.time)


def play_match(
    sim: TennisSim,
    home,
    away,
    seed: int,
    condition: ConditionConfig = NOMINAL,
    game: GameConfig | None = None,
    on_step: Callable[[StepRecord], None] | None = None,
    first_server: str = "home",
    stop_after_points: int | None = None,
) -> MatchResult:
    return Match(
        sim, home, away, seed, condition, game, on_step, first_server, stop_after_points
    ).run()
