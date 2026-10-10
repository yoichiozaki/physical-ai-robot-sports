"""Headless MuJoCo rendering with 3-D overlays of the robots' decisions.

Overlays are extra (non-physical) geoms appended to the ``mjvScene``: they are drawn in 3-D
with correct perspective and occlusion and only visualise real algorithm outputs
(``DecisionInfo``): the predicted ball path and bounces, the planned contact point, the
intended shot path and the aimed landing target.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field

import mujoco
import numpy as np

from tennis.control.players import DecisionInfo
from tennis.sim.kinematics import player_to_world
from tennis.sim.physics import TennisSim


@dataclass
class Overlay:
    kind: str  # "polyline" | "sphere" | "disc"
    points: list[tuple[float, float, float]] = field(default_factory=list)
    rgba: tuple[float, float, float, float] = (1.0, 1.0, 1.0, 1.0)
    size: float = 0.01  # line width / radius
    dashed: bool = False


COLORS = {
    "pred": (0.10, 0.85, 0.35, 0.90),
    "bounce": (0.05, 0.60, 0.25, 1.0),
    "contact": (1.00, 0.55, 0.05, 0.85),
    "shot": (1.00, 0.85, 0.10, 0.90),
    "target": (0.20, 0.45, 1.00, 0.60),
    "pred_away": (0.95, 0.35, 0.65, 0.75),
    "target_away": (1.00, 0.30, 0.25, 0.50),
}


def decision_overlays(info: DecisionInfo, player: str, ball_r: float = 0.033) -> list[Overlay]:
    w = lambda p: tuple(player_to_world(player, np.asarray(p, float)))  # noqa: E731
    home = player == "home"
    out: list[Overlay] = []
    if info.predicted_path and len(info.predicted_path) >= 2:
        out.append(
            Overlay("polyline", [w(p) for p in info.predicted_path],
                    COLORS["pred" if home else "pred_away"], 0.007)
        )  # fmt: skip
        for b in info.bounce_points:
            out.append(Overlay("disc", [w((b[0], b[1], 0.0))], COLORS["bounce"], 0.06))
    if info.shot_path and len(info.shot_path) >= 2:
        out.append(Overlay("polyline", [w(p) for p in info.shot_path], COLORS["shot"], 0.006, True))
    if info.contact is not None:
        out.append(Overlay("sphere", [w(info.contact)], COLORS["contact"], ball_r * 1.6))
    if info.target is not None:
        tx, ty = info.target
        out.append(
            Overlay("disc", [w((tx, ty, 0.0))], COLORS["target" if home else "target_away"], 0.12)
        )
    return out


class SceneRenderer:
    def __init__(self, sim: TennisSim, width: int = 1280, height: int = 720,
                 camera: str = "broadcast") -> None:  # fmt: skip
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
    def _add_geom(self, gtype: int, size, pos, rgba, mat=None) -> None:
        scn = self.renderer.scene
        if scn.ngeom >= scn.maxgeom:
            return
        g = scn.geoms[scn.ngeom]
        mujoco.mjv_initGeom(
            g, gtype, np.asarray(size, dtype=np.float64), np.asarray(pos, dtype=np.float64),
            (np.eye(3) if mat is None else mat).flatten().astype(np.float64),
            np.asarray(rgba, dtype=np.float32),
        )  # fmt: skip
        g.emission = 0.4
        scn.ngeom += 1

    def _add_segment(self, a, b, width: float, rgba) -> None:
        scn = self.renderer.scene
        if scn.ngeom >= scn.maxgeom:
            return
        g = scn.geoms[scn.ngeom]
        mujoco.mjv_initGeom(
            g, mujoco.mjtGeom.mjGEOM_CAPSULE, np.zeros(3), np.zeros(3), np.eye(3).flatten(),
            np.asarray(rgba, dtype=np.float32),
        )  # fmt: skip
        mujoco.mjv_connector(
            g, mujoco.mjtGeom.mjGEOM_CAPSULE, width,
            np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64),
        )  # fmt: skip
        g.emission = 0.4
        scn.ngeom += 1

    def add_overlay(self, ov: Overlay) -> None:
        if ov.kind == "polyline":
            pts = [np.asarray(p, float) for p in ov.points]
            for i, (a, b) in enumerate(itertools.pairwise(pts)):
                if not ov.dashed or i % 2 == 0:
                    self._add_segment(a, b, ov.size, ov.rgba)
        elif ov.kind == "sphere":
            for p in ov.points:
                self._add_geom(mujoco.mjtGeom.mjGEOM_SPHERE, (ov.size, 0, 0), p, ov.rgba)
        elif ov.kind == "disc":
            for p in ov.points:
                self._add_geom(
                    mujoco.mjtGeom.mjGEOM_CYLINDER,
                    (ov.size, 0.002, 0),
                    (p[0], p[1], 0.004),
                    ov.rgba,
                )

    # ------------------------------------------------------------------ render
    def render(
        self, overlays: list[Overlay] | None = None, camera: str | None = None
    ) -> np.ndarray:
        self.renderer.update_scene(
            self.sim.data, camera=camera or self.camera, scene_option=self.option
        )
        for ov in overlays or []:
            self.add_overlay(ov)
        return self.renderer.render().copy()

    def project(self, points: np.ndarray, camera: str | None = None) -> np.ndarray:
        """World points (N, 3) -> pixel coordinates (N, 2) for the named camera."""
        m, d = self.sim.model, self.sim.data
        cid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_CAMERA, camera or self.camera)
        pos = d.cam_xpos[cid]
        rot = d.cam_xmat[cid].reshape(3, 3)
        f = 0.5 * self.height / np.tan(np.deg2rad(m.cam_fovy[cid]) / 2)
        pc = (np.atleast_2d(points) - pos) @ rot  # camera coordinates (x right, y up, -z fwd)
        z = -pc[:, 2]
        return np.stack([self.width / 2 + f * pc[:, 0] / z, self.height / 2 - f * pc[:, 1] / z], 1)
