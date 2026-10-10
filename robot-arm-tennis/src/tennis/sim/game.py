"""Tennis rules (simplified singles) on top of the simulator's contact events.

* Each point starts with the server's *toss*: the ball is released above the server's
  court, must bounce once and is then hit (a "drop serve").
* After a hit the ball must cross the net and its first bounce must be inside the
  opponent's singles court (lines are in); the receiver may volley or play it after
  exactly one bounce. A second bounce, a ball hitting a robot's body, or a double hit
  loses the point. Servers alternate every point.
* Between points the ball is parked off court for ``point_gap`` seconds so that both robots
  can return to their ready pose before the next toss (without it, a robot pulled wide at
  the end of a point could not reach its own serve - an exploitable artefact).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from tennis.sim.physics import StepResult, TennisSim

PARK_POS = (0.0, 3.6, 0.034)  # beside the net post, out of every robot's reach


@dataclass(frozen=True)
class GameConfig:
    points: int = 20
    toss_x: float = -2.62  # toss position in the server's player frame
    toss_y: float = -0.10
    toss_z: float = 1.10
    toss_jitter: float = 0.12
    max_point_time: float = 30.0
    same_stroke_window: float = 0.15  # several racket contacts within this time = one stroke
    point_gap: float = 1.5  # [s] pause between points (ball parked, robots recover)
    # service box: the serve's first bounce must be within this distance of the net (tennis:
    # 6.40 m of 11.885 m -> 1.62 m here). None = whole court (the original, exploitable rule)
    service_line: float | None = None


@dataclass
class PointResult:
    server: str
    winner: str
    reason: str
    hits: int
    duration: float
    start: float
    end: float


OTHER = {"home": "away", "away": "home"}


@dataclass
class Referee:
    sim: TennisSim
    cfg: GameConfig
    rng: np.random.Generator
    score: dict[str, int] = field(default_factory=lambda: {"home": 0, "away": 0})
    points: list[PointResult] = field(default_factory=list)
    server: str = "home"
    last_hitter: str | None = None
    last_hit_time: float = -10.0
    bounces: dict[str, int] = field(default_factory=lambda: {"home": 0, "away": 0})
    hits: int = 0
    point_start: float = 0.0
    live: bool = False
    ball_id: int = 0
    shot_log: list[dict] = field(default_factory=list)  # every hit and where it bounced first

    def start_point(self, server: str | None = None) -> None:
        self.server = server or self.server
        c, j = self.cfg, self.cfg.toss_jitter
        p = np.array(
            [c.toss_x + self.rng.uniform(-j, j), c.toss_y + self.rng.uniform(-j, j), c.toss_z]
        )
        if self.server == "away":
            p = p * np.array([-1.0, -1.0, 1.0])
        self.sim.place_ball(p, (0.0, 0.0, 0.0))
        self.last_hitter = None
        self.last_hit_time = -10.0
        self.bounces = {"home": 0, "away": 0}
        self.hits = 0
        self.point_start = self.sim.time
        self.live = True
        self.ball_id += 1

    def _end(self, winner: str, reason: str) -> PointResult:
        t = self.sim.time
        res = PointResult(
            self.server, winner, reason, self.hits, t - self.point_start, self.point_start, t
        )
        self.points.append(res)
        self.score[winner] += 1
        self.live = False
        self.server = OTHER[self.server]
        return res

    def update(self, step: StepResult) -> PointResult | None:
        if not self.live:
            return None
        hw = self.sim.cfg.court.width / 2 + self.sim.cfg.court.line_width / 2
        hl = self.sim.cfg.court.length / 2 + self.sim.cfg.court.line_width / 2
        for ev in step.events:
            kind = ev.kind
            if kind.startswith("racket_"):
                p = kind.split("_", 1)[1]
                if (
                    self.last_hitter == p
                    and ev.time - self.last_hit_time < self.cfg.same_stroke_window
                ):
                    continue
                if self.last_hitter == p:
                    return self._end(OTHER[p], "double hit")
                if self.last_hitter is None and p != self.server:
                    continue
                if (
                    self.shot_log
                    and self.shot_log[-1]["player"] == OTHER[p]
                    and self.shot_log[-1]["landing"] is None
                ):
                    self.shot_log[-1]["landing"] = "volleyed"
                self.last_hitter, self.last_hit_time = p, ev.time
                self.bounces = {"home": 0, "away": 0}
                self.hits += 1
                self.ball_id += 1
                self.shot_log.append(
                    {
                        "player": p,
                        "time": ev.time,
                        "landing": None,
                        "pos": None,
                        "serve": self.hits == 1,
                    }
                )
            elif kind.startswith("body_"):
                q = kind.split("_", 1)[1]
                return self._end(OTHER[q], f"ball hit {q} robot")
            elif kind == "floor":
                x, y = float(ev.ball_pos[0]), float(ev.ball_pos[1])
                side = "home" if x < 0 else "away"
                if self.last_hitter is None:
                    if side != self.server:
                        return self._end(OTHER[self.server], "toss error")
                    self.bounces[side] += 1
                    if self.bounces[side] >= 2:
                        return self._end(OTHER[self.server], "serve not hit")
                    continue
                p, o = self.last_hitter, OTHER[self.last_hitter]
                if side == p:
                    self._log_landing("net/short", x, y)
                    return self._end(o, "did not cross the net")
                if self.bounces[o] == 0:
                    inside = abs(y) <= hw and abs(x) <= hl
                    sl = self.cfg.service_line
                    serve = self.shot_log[-1]["serve"] if self.shot_log else False
                    if (
                        inside
                        and serve
                        and sl is not None
                        and abs(x) > sl + self.sim.cfg.court.line_width / 2
                    ):
                        self._log_landing("out", x, y)
                        return self._end(o, "serve out of the service box")
                    self._log_landing("in" if inside else "out", x, y)
                    if not inside:
                        return self._end(o, "out")
                    self.bounces[o] = 1
                else:
                    return self._end(p, "not returned")
        if step.lost:
            if self.last_hitter is None:
                return self._end(OTHER[self.server], "toss lost")
            p, o = self.last_hitter, OTHER[self.last_hitter]
            if self.bounces[o] == 0:
                self._log_landing("out", None, None)
                return self._end(o, "out")
            return self._end(p, "not returned")
        if self.sim.time - self.point_start > self.cfg.max_point_time:
            if self.last_hitter is None:
                return self._end(OTHER[self.server], "serve timeout")
            p, o = self.last_hitter, OTHER[self.last_hitter]
            return self._end(p if self.bounces[o] else o, "timeout")
        return None

    def _log_landing(self, kind: str, x: float | None, y: float | None) -> None:
        if self.shot_log and self.shot_log[-1]["landing"] is None:
            self.shot_log[-1]["landing"] = kind
            self.shot_log[-1]["pos"] = None if x is None else (x, y)

    def bounced_on(self, player: str) -> bool:
        return self.bounces[player] > 0
