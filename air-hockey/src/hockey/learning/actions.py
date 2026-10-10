"""Action interfaces between learned policies and the robot.

Both adapters end in a *target position* for the mallet, which the simulator's
``MotionController`` turns into limited joint velocities: learned actions always pass
through the robot's physical constraints.

* ``DirectTargetAction`` (pure RL): Box(-1, 1, 2) mapped onto the robot workspace.
* ``HybridStrategyAction`` (physics + learning): MultiDiscrete([strategy, aim, speed]);
  the physics-based controller executes the chosen strategy (prediction, interception
  timing, shot geometry). The policy also receives physics-derived features.
"""

from __future__ import annotations

import numpy as np
from gymnasium import spaces

from hockey.controllers.base import Command, DecisionInfo
from hockey.controllers.physics_based import PhysicsController, PhysicsControllerConfig
from hockey.controllers.planning import ShotSpec, Strategy, path_clearance, shot_path
from hockey.controllers.prediction import PuckModel
from hockey.observation import Geometry, PlayerView

DEFAULT_AIMS = (-0.09, -0.045, 0.0, 0.045, 0.09)
DEFAULT_SPEEDS = (2.5, 3.5, 4.5)


class DirectTargetAction:
    name = "direct"
    feature_dim = 0

    def __init__(self, geom: Geometry) -> None:
        self.geom = geom
        self.space = spaces.Box(-1.0, 1.0, shape=(2,), dtype=np.float32)
        self._target = np.array([geom.home_x, 0.0])

    def reset(self, rng: np.random.Generator) -> None:
        self._target = np.array([self.geom.home_x, 0.0])

    def decode(self, action: np.ndarray) -> np.ndarray:
        a = np.clip(np.asarray(action, dtype=float).reshape(2), -1.0, 1.0)
        (x0, x1), (y0, y1) = self.geom.x_range, self.geom.y_range
        return np.array([x0 + (a[0] + 1) * 0.5 * (x1 - x0), y0 + (a[1] + 1) * 0.5 * (y1 - y0)])

    def set_action(self, action: np.ndarray) -> None:
        self._target = self.decode(action)

    def command(self, view: PlayerView) -> Command:
        return Command(self._target.copy(), DecisionInfo(mode="policy", strategy="RL target"))

    def needs_decision(self, view: PlayerView) -> bool:  # fixed-rate decisions
        return True

    def features(self, view: PlayerView) -> np.ndarray:
        return np.zeros(0, dtype=np.float32)


class HybridStrategyAction:
    name = "hybrid"
    feature_dim = 12

    def __init__(
        self,
        geom: Geometry,
        model: PuckModel | None = None,
        aims: tuple[float, ...] = DEFAULT_AIMS,
        speeds: tuple[float, ...] = DEFAULT_SPEEDS,
        executor_cfg: PhysicsControllerConfig | None = None,
    ) -> None:
        self.geom = geom
        self.aims = aims
        self.speeds = speeds
        self.executor = PhysicsController(
            geom, executor_cfg or PhysicsControllerConfig(name="Hybrid"), model
        )
        self.space = spaces.MultiDiscrete([len(Strategy), len(aims), len(speeds)])
        self.executor.forced_spec = ShotSpec(Strategy.DIRECT, 0.0, speeds[len(speeds) // 2])

    def reset(self, rng: np.random.Generator) -> None:
        self.executor.reset(rng)
        self.executor.forced_spec = ShotSpec(
            Strategy.DIRECT, 0.0, self.speeds[len(self.speeds) // 2]
        )

    def decode(self, action: np.ndarray) -> ShotSpec:
        a = np.asarray(action, dtype=int).reshape(3)
        return ShotSpec(
            Strategy(int(a[0])), float(self.aims[int(a[1])]), float(self.speeds[int(a[2])])
        )

    def set_action(self, action: np.ndarray) -> None:
        self.executor.forced_spec = self.decode(action)

    def needs_decision(self, view: PlayerView) -> bool:
        """Event-driven decisions: ask the policy for a new shot only when the executor is
        idle (previous shot finished / no committed plan) and an attack is possible."""
        return self.executor.idle and (view.puck_pos[0] < 0.3 or view.puck_vel[0] < -0.1)

    def command(self, view: PlayerView) -> Command:
        return self.executor.act(view)

    def features(self, view: PlayerView) -> np.ndarray:
        """Physics-derived features: defence crossing, earliest strike, lane clearances."""
        g = self.geom
        ex = self.executor
        traj = ex.predictor.predict(view.puck_pos, view.puck_vel, ex.cfg.horizon)
        f = np.zeros(self.feature_dim, dtype=np.float32)
        x_def = -g.half_length + ex.cfg.defense_offset
        hit = traj.first_crossing_x(x_def) if view.puck_vel[0] < -0.05 else None
        if hit is not None:
            f[0], f[1], f[2] = 1.0, min(hit[0], 1.5) / 1.5, hit[1] / g.half_width
        spec = ShotSpec(Strategy.DIRECT, 0.0, self.speeds[len(self.speeds) // 2])
        plan = ex.planner.find_strike(
            traj, spec, view.own_pos, view.own_vel, ex.limits, ex.cfg.horizon
        )
        if plan is not None:
            px, py = plan.puck_contact
            f[3], f[4] = 1.0, min(plan.t_contact, 1.5) / 1.5
            f[5], f[6] = px / g.half_length, py / g.half_width
            ox, oy = float(view.opp_pos[0]), float(view.opp_pos[1])
            for i, strat in enumerate((Strategy.DIRECT, Strategy.BANK_LEFT, Strategy.BANK_RIGHT)):
                path = shot_path(
                    px, py, ShotSpec(strat, 0.0, spec.speed), g, ex.model.wall_restitution
                )
                f[7 + i] = min(path_clearance(path, ox, oy), 0.5) / 0.5
            for i, strat in enumerate((Strategy.BANK_LEFT, Strategy.BANK_RIGHT)):
                alt = ex.planner.plan_at(
                    plan.t_contact,
                    plan.puck_contact,
                    plan.puck_vel,
                    ShotSpec(strat, 0.0, spec.speed),
                    view.own_pos,
                    view.own_vel,
                    ex.limits,
                )
                f[10 + i] = float(alt is not None and alt.slack >= 0.0)
        return f
