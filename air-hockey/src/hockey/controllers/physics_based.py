"""Baseline B: physics-based predictive controller (prediction + interception + attack planning).

Decision cycle (every control step, 50 Hz):
  1. Predict the puck trajectory with decay and wall reflections (``PuckPredictor``).
  2. Search the earliest *feasible* strike on that trajectory: the robot must reach the
     run-up (staging) point and accelerate through the contact point before the puck
     arrives (robot travel time vs puck arrival time). Unreachable impacts are discarded.
  3. Execute: move to the staging point, launch the strike when the timing slack reaches
     zero, keep the strike committed until the impact window passes.
  4. Without a feasible strike: block on the defence line where the predicted path
     crosses it, or retreat to a guarding position.

The same class, with different ``PhysicsControllerConfig`` values, provides the scripted
"defensive" and "aggressive" opponents, and the execution layer of the hybrid AI
(``forced_spec`` is set by the learned strategy selector).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, replace

import numpy as np

from hockey.controllers.adaptation import OnlinePuckEstimator, PuckFilter
from hockey.controllers.base import Command, DecisionInfo
from hockey.controllers.planning import (
    STRATEGY_LABELS,
    PlannerConfig,
    ShotSpec,
    Strategy,
    StrikePlan,
    StrikePlanner,
    defense_point,
    detour_target,
    path_clearance,
    shot_path,
)
from hockey.controllers.prediction import FORECAST_HORIZON, PuckModel, PuckPredictor
from hockey.observation import Geometry, PlayerView


@dataclass(frozen=True)
class PhysicsControllerConfig:
    name: str = "Physics"
    shot_selection: str = "direct"  # "direct" | "heuristic"
    shot_speed: float = 3.5
    aim_y: float = 0.0
    aim_jitter: float = 0.0  # uniform random aim offset (scripted opponents)
    attack_moving: bool = True  # strike moving pucks
    attack_slow: bool = True  # strike slow/stationary pucks in our half
    slow_speed: float = 0.6
    defense_offset: float = 0.13  # defence line distance from own goal line
    retreat_offset: float | None = None  # guarding x distance from own goal line (None: home)
    retreat_y_gain: float = 0.35
    horizon: float = 1.5
    min_slack: float = 0.03  # timing margin required when (re)selecting an impact point
    commit_window: float = 0.12
    launch_reach_tol: float = 0.04  # max time-to-staging allowed when launching a strike
    launch_dist: float = 0.03  # also launch when within this distance of the staging point
    commit_tolerance: float = 0.06  # keep a committed impact time while slack >= -this [s]
    still_speed: float = 0.1  # filtered puck speeds below this are treated as zero (noise deadband)
    filter_measurements: bool = True  # alpha-beta filtering of the puck measurement
    planner: PlannerConfig = field(default_factory=PlannerConfig)


class PhysicsController:
    def __init__(
        self,
        geom: Geometry,
        cfg: PhysicsControllerConfig | None = None,
        model: PuckModel | None = None,
        adaptive: bool = False,
    ) -> None:
        self.geom = geom
        self.cfg = cfg or PhysicsControllerConfig()
        self.name = self.cfg.name
        self.model = model or PuckModel()
        self.prior_model = self.model
        self.estimator = OnlinePuckEstimator(geom, self.model) if adaptive else None
        self.filter = PuckFilter(self.model) if self.cfg.filter_measurements else None
        self.predictor = PuckPredictor(geom, self.model)
        self.planner = StrikePlanner(
            geom,
            replace(
                self.cfg.planner,
                mallet_restitution=self.model.mallet_restitution,
                wall_restitution=self.model.wall_restitution,
            ),
        )
        self.limits = geom.axis_limits()
        self.forced_spec: ShotSpec | None = None
        self.rng = np.random.default_rng(0)
        self._strike: StrikePlan | None = None
        self._strike_t0 = 0.0
        self._strike_start = 0.0
        self._strike_until = -1.0
        self._strike_target: np.ndarray | None = None
        self._commit_t: float | None = None
        self._commit_spec: ShotSpec | None = None
        self._heur_spec: ShotSpec | None = None
        self._heur_until = -1.0
        self._aim_offset = 0.0
        self.last_plan: StrikePlan | None = None

    # ------------------------------------------------------------------ setup
    def reset(self, rng: np.random.Generator) -> None:
        self.rng = rng
        self._strike = None
        self._strike_until = -1.0
        self._commit_t = None
        self._commit_spec = None
        self._heur_spec = None
        self._heur_until = -1.0
        self._aim_offset = 0.0
        self.last_plan = None
        if self.estimator is not None:
            self.estimator.reset()
            self.set_model(self.prior_model)
        if self.filter is not None:
            self.filter.reset()

    def set_speed_scale(self, scale: float) -> None:
        self.limits = self.geom.axis_limits(scale)

    @property
    def idle(self) -> bool:
        """No strike in progress and no committed impact plan (a new shot can be chosen)."""
        return self._strike is None and self._commit_t is None

    def set_model(self, model: PuckModel) -> None:
        """Update the physical model (e.g. from online parameter estimation)."""
        self.model = model
        self.predictor.set_model(model)
        self.planner.cfg = replace(
            self.planner.cfg,
            mallet_restitution=model.mallet_restitution,
            wall_restitution=model.wall_restitution,
        )

    # ------------------------------------------------------------------ helpers
    def _guard_target(self, view: PlayerView) -> np.ndarray:
        g = self.geom
        off = self.cfg.retreat_offset
        x = g.home_x if off is None else -g.half_length + off
        lim = g.goal_width / 2 + g.mallet_radius
        y = float(np.clip(self.cfg.retreat_y_gain * view.puck_pos[1], -lim, lim))
        return np.array([x, y])

    def _base_spec(self) -> ShotSpec:
        if self.forced_spec is not None:
            return self.forced_spec
        return ShotSpec(Strategy.DIRECT, self.cfg.aim_y + self._aim_offset, self.cfg.shot_speed)

    def _heuristic_spec(self, plan: StrikePlan, view: PlayerView) -> ShotSpec:
        """Pick the candidate shot whose path stays farthest from the opponent mallet."""
        px, py = plan.puck_contact
        t = plan.t_contact
        ox = float(view.opp_pos[0] + 0.5 * view.opp_vel[0] * min(t, 0.3))
        oy = float(view.opp_pos[1] + 0.5 * view.opp_vel[1] * min(t, 0.3))
        best, best_c = plan.spec, -1.0
        g2 = self.geom.goal_width / 2 - self.geom.puck_radius - 0.01
        cands = [
            ShotSpec(Strategy.DIRECT, a, self.cfg.shot_speed) for a in (-g2 * 0.7, 0.0, g2 * 0.7)
        ]
        cands += [
            ShotSpec(s, 0.0, self.cfg.shot_speed) for s in (Strategy.BANK_LEFT, Strategy.BANK_RIGHT)
        ]
        for spec in cands:
            path = shot_path(px, py, spec, self.geom, self.model.wall_restitution)
            c = path_clearance(path, ox, oy)
            if spec.strategy != Strategy.DIRECT:
                c -= 0.03  # bank shots lose speed at the wall: prefer direct when similar
            if c > best_c:
                best, best_c = spec, c
        return best

    # ------------------------------------------------------------------ main
    def act(self, view: PlayerView) -> Command:
        g, cfg = self.geom, self.cfg
        if self.estimator is not None and self.estimator.update(view):  # raw measurements
            self.set_model(self.estimator.model)
        if self.filter is not None:
            self.filter.model = self.model
            p, v = self.filter.update(view.time, view.puck_pos, view.puck_vel)
            if float(np.hypot(*v)) < cfg.still_speed:
                v = np.zeros(2)
            view = replace(view, puck_pos=p, puck_vel=v)
        traj = self.predictor.predict(view.puck_pos, view.puck_vel, cfg.horizon)
        end = traj.state_at(traj.end_time)
        fx, fy, _, _ = traj.state_at(FORECAST_HORIZON)
        info = DecisionInfo(
            predicted_path=[
                (float(view.puck_pos[0]), float(view.puck_pos[1])),
                *traj.bounce_points(),
                end[:2],
            ],
            bounce_points=traj.bounce_points(),
            extra={"forecast_x": fx, "forecast_y": fy},
        )
        own_pos = view.own_pos
        puck_speed = float(np.hypot(*view.puck_vel))
        spec = self._base_spec()

        # 1) committed strike
        if self._strike is not None:
            hit_happened = (
                view.puck_vel[0] > 0.4 and view.puck_pos[0] > self._strike.puck_contact[0] - 0.05
            )
            if view.time <= self._strike_until and not hit_happened:
                plan = self._strike
                self._fill_plan_info(info, plan, view, self._strike_t0)
                info.mode = "strike"
                wait = self._strike_start - view.time
                if wait > 1e-3:  # launch still pending (time-stamped switch)
                    return Command(
                        np.array(plan.staging),
                        info,
                        switch_target=np.array(plan.strike_target),
                        switch_delay=wait,
                    )
                return Command(self._strike_line_target(plan, own_pos), info)
            self._strike = None

        # 2) attack planning
        plan = None
        attack_ok = spec.strategy != Strategy.DEFEND and (
            (cfg.attack_moving and puck_speed >= cfg.slow_speed)
            or (cfg.attack_slow and puck_speed < cfg.slow_speed)
        )
        if attack_ok and view.puck_pos[0] < g.half_length * 0.6:
            samples = traj.sample(self.planner.cfg.sample_dt, cfg.horizon)
            if self.forced_spec is None and cfg.shot_selection == "heuristic":
                spec = self._choose_heuristic_spec(traj, samples, view, spec)
            plan = self._plan_with_commitment(traj, samples, spec, view)
        self.last_plan = plan
        if plan is not None:
            self._fill_plan_info(info, plan, view, view.time)
            delay = plan.t_contact - plan.t_strike  # when the run-up must start
            at_staging = plan.t_reach <= cfg.launch_reach_tol or (
                cfg.launch_dist > 0
                and float(np.hypot(own_pos[0] - plan.staging[0], own_pos[1] - plan.staging[1]))
                <= cfg.launch_dist
                and float(np.hypot(*view.own_vel)) <= 0.5
            )
            if at_staging and delay <= g.control_dt + 1e-9:
                self._strike = plan
                self._strike_t0 = view.time
                self._strike_start = view.time + max(delay, 0.0)
                self._strike_until = view.time + plan.t_contact + cfg.commit_window
                self._strike_target = None
                self._commit_t = None
                self._heur_spec = None
                if cfg.aim_jitter > 0:
                    self._aim_offset = float(self.rng.uniform(-cfg.aim_jitter, cfg.aim_jitter))
                info.mode = "strike"
                if delay > 1e-3:
                    return Command(
                        np.array(plan.staging),
                        info,
                        switch_target=np.array(plan.strike_target),
                        switch_delay=delay,
                    )
                return Command(self._strike_line_target(plan, own_pos), info)
            target = plan.staging
            detour = detour_target(
                (float(own_pos[0]), float(own_pos[1])),
                plan.staging,
                (float(view.puck_pos[0]), float(view.puck_pos[1])),
                plan.normal,
                g.puck_radius + g.mallet_radius + 0.015,
            )
            info.mode = "position"
            return Command(np.array(detour if detour is not None else target), info)

        # 3) defence / guard
        x_def = -g.half_length + cfg.defense_offset
        # Block on the defence line, or "in place" if the mallet already stands further
        # forward between the puck and the goal (never retreat through the puck's path).
        own_x = float(own_pos[0])
        x_block = x_def
        if own_x > x_def and own_x < float(view.puck_pos[0]) - g.puck_radius - g.mallet_radius:
            x_block = own_x
        dp = defense_point(traj, g, x_block) if view.puck_vel[0] < -0.05 else None
        if dp is None and x_block != x_def and view.puck_vel[0] < -0.05:
            x_block = x_def
            dp = defense_point(traj, g, x_block)
        info.strategy = STRATEGY_LABELS[Strategy.DEFEND]
        if dp is not None:
            t_cross, y_cross = dp
            lim = g.goal_width / 2 + g.mallet_radius * 0.5
            y_block = float(np.clip(y_cross, -lim, lim)) if x_block == x_def else y_cross
            info.mode = "defend"
            info.intercept = (x_block, y_cross)
            info.intercept_time = t_cross
            return Command(np.array([x_block - 0.01, y_block]), info)
        info.mode = "guard"
        return Command(self._guard_target(view), info)

    def _choose_heuristic_spec(
        self, traj, samples: np.ndarray, view: PlayerView, base: ShotSpec
    ) -> ShotSpec:
        """Rule-based shot selection (open-lane heuristic), kept for 0.3 s to avoid dithering."""
        if self._heur_spec is not None and view.time < self._heur_until:
            return self._heur_spec
        prov = self.planner.find_strike(
            traj, base, view.own_pos, view.own_vel, self.limits, self.cfg.horizon, samples=samples
        )
        spec = base
        if prov is not None:
            cand = self._heuristic_spec(prov, view)
            if cand != base:
                alt = self.planner.plan_at(
                    prov.t_contact, prov.puck_contact, prov.puck_vel, cand,
                    view.own_pos, view.own_vel, self.limits,
                )  # fmt: skip
                if alt is not None and alt.slack >= 0.0:
                    spec = cand
        self._heur_spec, self._heur_until = spec, view.time + 0.3
        return spec

    def _plan_with_commitment(
        self, traj, samples: np.ndarray, spec: ShotSpec, view: PlayerView
    ) -> StrikePlan | None:
        """Keep the previously chosen impact time while it stays feasible (hysteresis).

        Re-selecting the earliest feasible impact every cycle makes the staging point chase a
        moving target; committing to an absolute impact time lets the robot settle.
        """
        own_pos, own_vel = view.own_pos, view.own_vel
        if self._commit_t is not None and self._commit_spec == spec:
            t_rel = self._commit_t - view.time
            if t_rel > 0.0:
                x, y, vx, vy = traj.state_at(t_rel)
                plan = self.planner.plan_at(
                    t_rel, (x, y), (vx, vy), spec, own_pos, own_vel, self.limits
                )
                # Tolerate small timing deficits: measurement noise moves the staging point by
                # millimetres and the strike line is re-aimed from the actual mallet position.
                if plan is not None and plan.slack >= -self.cfg.commit_tolerance:
                    return plan
        plan = self.planner.find_strike(
            traj,
            spec,
            own_pos,
            own_vel,
            self.limits,
            self.cfg.horizon,
            min_slack=self.cfg.min_slack,
            samples=samples,
        )
        if plan is None:  # nothing with a comfortable margin: accept any feasible strike
            plan = self.planner.find_strike(
                traj, spec, own_pos, own_vel, self.limits, self.cfg.horizon, samples=samples
            )
        self._commit_t = None if plan is None else view.time + plan.t_contact
        self._commit_spec = spec
        return plan

    def _strike_line_target(self, plan: StrikePlan, own_pos: np.ndarray) -> np.ndarray:
        """Aim the straight-line strike so the mallet centre passes exactly through the planned
        contact point c (which fixes the contact normal), whatever small offset remains between
        the mallet and the staging point. Frozen once the mallet is within 4 cm of c."""
        cx, cy = plan.mallet_contact
        dx, dy = cx - float(own_pos[0]), cy - float(own_pos[1])
        dist = float(np.hypot(dx, dy))
        if self._strike_target is not None and dist < 0.04:
            return self._strike_target
        nx, ny = plan.normal
        if dist < 1e-6 or (dx * nx + dy * ny) / dist < 0.5:
            ux, uy = nx, ny  # too far off the strike line: fall back to the planned direction
        else:
            ux, uy = dx / dist, dy / dist
        follow = float(np.hypot(plan.strike_target[0] - cx, plan.strike_target[1] - cy))
        self._strike_target = np.array([cx + ux * follow, cy + uy * follow])
        return self._strike_target

    def _fill_plan_info(
        self, info: DecisionInfo, plan: StrikePlan, view: PlayerView, t0: float
    ) -> None:
        info.strategy = STRATEGY_LABELS[plan.spec.strategy]
        info.intercept = plan.puck_contact
        info.intercept_time = max(plan.t_contact - (view.time - t0), 0.0)
        info.mallet_contact = plan.mallet_contact
        info.shot_path = shot_path(
            *plan.puck_contact, plan.spec, self.geom, self.model.wall_restitution
        )
        info.extra.update(
            {
                "slack": plan.slack,
                "mallet_speed": plan.mallet_speed,
                "expected_out_speed": plan.expected_out_speed,
                "aim_y": plan.spec.aim_y,
            }
        )


def strike_direction_error(plan: StrikePlan, out_vel: tuple[float, float]) -> float:
    """Angle [rad] between the planned and realised outgoing puck direction."""
    a = math.atan2(plan.direction[1], plan.direction[0])
    b = math.atan2(out_vel[1], out_vel[0])
    d = (b - a + math.pi) % (2 * math.pi) - math.pi
    return abs(d)
