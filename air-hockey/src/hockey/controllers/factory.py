"""Build controllers from short string specs used by scripts and configs.

Specs:
    reactive | physics | physics_heuristic            our baselines (A, B, B+)
    ppo:<path/to/model.zip> | hybrid:<path/to/model.zip>   learned controllers (C, D)
    stationary | random | defensive | aggressive      scripted opponents
"""

from __future__ import annotations

from hockey.controllers.base import Controller
from hockey.controllers.opponents import make_opponent
from hockey.controllers.physics_based import PhysicsController, PhysicsControllerConfig
from hockey.controllers.prediction import PuckModel
from hockey.controllers.reactive import ReactiveController
from hockey.observation import Geometry

DISPLAY_NAMES = {
    "reactive": "Reactive",
    "physics": "Physics",
    "physics_heuristic": "Physics+Rule",
}


def make_controller(spec: str, geom: Geometry, model: PuckModel | None = None) -> Controller:
    kind, _, arg = spec.partition(":")
    kind = kind.lower()
    if kind == "reactive":
        return ReactiveController(geom)
    if kind == "physics":
        return PhysicsController(geom, PhysicsControllerConfig(name="Physics"), model)
    if kind == "physics_adaptive":
        return PhysicsController(
            geom, PhysicsControllerConfig(name="Physics+Adapt"), model, adaptive=True
        )
    if kind == "physics_heuristic":
        return PhysicsController(
            geom, PhysicsControllerConfig(name="Physics+Rule", shot_selection="heuristic"), model
        )
    if kind in ("ppo", "hybrid", "hybrid_adaptive"):
        from hockey.controllers.learned import PolicyController

        names = {
            "ppo": "PPO (pure RL)",
            "hybrid": "Hybrid AI",
            "hybrid_adaptive": "Hybrid AI+Adapt",
        }
        return PolicyController(
            arg, geom, model, name=names[kind], adaptive=kind == "hybrid_adaptive"
        )
    return make_opponent(kind, geom, model)
