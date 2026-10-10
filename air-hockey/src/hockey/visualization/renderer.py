"""Headless MuJoCo rendering with in-scene overlays of controller decisions.

Overlays are drawn as extra (non-physical) geoms appended to the ``mjvScene`` so they are
rendered in 3-D with correct perspective/occlusion. They only visualise real algorithm
outputs (``DecisionInfo``): predicted puck path, wall bounces, planned impact point,
intended shot path and the robot's commanded target.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field

import mujoco
import numpy as np

from hockey.controllers.base import DecisionInfo
from hockey.observation import to_world
from hockey.simulation.physics import HockeySim

Z_OVERLAY = 0.004


@dataclass
class Overlay:
    kind: str  # "polyline" | "disc" | "sphere" | "arrow"
    points: list[tuple[float, float]] = field(default_factory=list)
    rgba: tuple[float, float, float, float] = (1.0, 1.0, 1.0, 1.0)
    size: float = 0.01  # line width / radius
    z: float = Z_OVERLAY
    dashed: bool = False


COLORS = {
    "pred": (0.10, 0.85, 0.35, 0.95),
    "bounce": (0.05, 0.60, 0.25, 1.0),
    "intercept": (1.00, 0.55, 0.05, 0.85),
    "contact": (0.15, 0.85, 1.00, 0.55),
    "shot": (1.00, 0.85, 0.10, 0.90),
    "target": (0.20, 0.45, 1.00, 0.45),
    "target_away": (1.00, 0.30, 0.25, 0.40),
    "pred_away": (0.95, 0.35, 0.65, 0.80),
}


def decision_overlays(
    info: DecisionInfo, player: str, target: np.ndarray | None, puck_r: float, mallet_r: float
) -> list[Overlay]:
    w = lambda p: tuple(to_world(player, p))  # noqa: E731
    out: list[Overlay] = []
    home = player == "home"
    if info.predicted_path and len(info.predicted_path) >= 2:
        out.append(
            Overlay(
                "polyline",
                [w(p) for p in info.predicted_path],
                COLORS["pred" if home else "pred_away"],
                0.006,
            )
        )
        for b in info.bounce_points:
            out.append(Overlay("sphere", [w(b)], COLORS["bounce"], 0.012, z=0.01))
    if info.shot_path and len(info.shot_path) >= 2:
        out.append(
            Overlay("polyline", [w(p) for p in info.shot_path], COLORS["shot"], 0.004, dashed=True)
        )
    if info.intercept is not None:
        out.append(
            Overlay("disc", [w(info.intercept)], COLORS["intercept"], puck_r * 1.15, z=0.002)
        )
    if info.mallet_contact is not None:
        out.append(Overlay("disc", [w(info.mallet_contact)], COLORS["contact"], mallet_r, z=0.0015))
    if target is not None:
        out.append(
            Overlay(
                "disc",
                [w(target)],
                COLORS["target" if home else "target_away"],
                mallet_r * 0.45,
                z=0.003,
            )
        )
    return out


class SceneRenderer:
    def __init__(
        self, sim: HockeySim, width: int = 1280, height: int = 720, camera: str = "broadcast"
    ) -> None:
        self.sim = sim
        self.width, self.height = width, height
        self.camera = camera
        self.renderer = mujoco.Renderer(sim.model, height, width)
        self.option = mujoco.MjvOption()

    def close(self) -> None:
        if self.renderer is not None:
            self.renderer.close()
            self.renderer = None

    def __enter__(self) -> SceneRenderer:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    # ------------------------------------------------------------------ overlays
    def _add_geom(self, gtype: int, size, pos, rgba, mat=None) -> mujoco.MjvGeom | None:
        scn = self.renderer.scene
        if scn.ngeom >= scn.maxgeom:
            return None
        g = scn.geoms[scn.ngeom]
        mujoco.mjv_initGeom(
            g,
            gtype,
            np.asarray(size, dtype=np.float64),
            np.asarray(pos, dtype=np.float64),
            (np.eye(3) if mat is None else mat).flatten().astype(np.float64),
            np.asarray(rgba, dtype=np.float32),
        )
        g.emission = 0.35
        scn.ngeom += 1
        return g

    def _add_segment(self, a, b, width: float, rgba) -> None:
        scn = self.renderer.scene
        if scn.ngeom >= scn.maxgeom:
            return
        g = scn.geoms[scn.ngeom]
        mujoco.mjv_initGeom(
            g,
            mujoco.mjtGeom.mjGEOM_CAPSULE,
            np.zeros(3),
            np.zeros(3),
            np.eye(3).flatten(),
            np.asarray(rgba, dtype=np.float32),
        )
        mujoco.mjv_connector(
            g,
            mujoco.mjtGeom.mjGEOM_CAPSULE,
            width,
            np.asarray(a, dtype=np.float64),
            np.asarray(b, dtype=np.float64),
        )
        g.emission = 0.35
        scn.ngeom += 1

    def add_overlay(self, ov: Overlay) -> None:
        z = ov.z
        if ov.kind == "polyline":
            pts = [np.array([p[0], p[1], z]) for p in ov.points]
            for a, b in itertools.pairwise(pts):
                if ov.dashed:
                    n = max(1, int(np.linalg.norm(b - a) / 0.05))
                    for i in range(n):
                        if i % 2 == 0:
                            self._add_segment(
                                a + (b - a) * i / n, a + (b - a) * (i + 1) / n, ov.size, ov.rgba
                            )
                else:
                    self._add_segment(a, b, ov.size, ov.rgba)
        elif ov.kind == "disc":
            for p in ov.points:
                self._add_geom(
                    mujoco.mjtGeom.mjGEOM_CYLINDER, (ov.size, 0.0015, 0), (p[0], p[1], z), ov.rgba
                )
        elif ov.kind == "sphere":
            for p in ov.points:
                self._add_geom(
                    mujoco.mjtGeom.mjGEOM_SPHERE, (ov.size, 0, 0), (p[0], p[1], z), ov.rgba
                )
        elif ov.kind == "arrow" and len(ov.points) >= 2:
            self._add_segment((*ov.points[0], z), (*ov.points[1], z), ov.size, ov.rgba)

    # ------------------------------------------------------------------ render
    def render(
        self, overlays: list[Overlay] | None = None, camera: str | None = None
    ) -> np.ndarray:
        cam = camera or self.camera
        self.renderer.update_scene(self.sim.data, camera=cam, scene_option=self.option)
        for ov in overlays or []:
            self.add_overlay(ov)
        return self.renderer.render().copy()

    def project(self, points: np.ndarray, camera: str | None = None) -> np.ndarray:
        """World points (N, 3) -> pixel coordinates (N, 2) for the named camera."""
        m, d = self.sim.model, self.sim.data
        cid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_CAMERA, camera or self.camera)
        pos = d.cam_xpos[cid]
        rot = d.cam_xmat[cid].reshape(3, 3)
        fovy = np.deg2rad(m.cam_fovy[cid])
        f = 0.5 * self.height / np.tan(fovy / 2)
        pc = (np.atleast_2d(points) - pos) @ rot  # camera coordinates (x right, y up, -z forward)
        z = -pc[:, 2]
        u = self.width / 2 + f * pc[:, 0] / z
        v = self.height / 2 - f * pc[:, 1] / z
        return np.stack([u, v], axis=1)
