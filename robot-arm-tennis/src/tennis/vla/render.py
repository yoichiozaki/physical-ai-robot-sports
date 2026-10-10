"""Camera images for vision policies (offscreen MuJoCo rendering, no overlays)."""

from __future__ import annotations

import mujoco
import numpy as np

from tennis.sim.physics import TennisSim


class PolicyCamera:
    """Callable ``camera(sim) -> uint8 image (H, W, 3)`` for :func:`tennis.vla.episode.run_episode`.

    One renderer per simulator (worlds have separate MuJoCo models)."""

    def __init__(self, size: int = 256, camera: str = "vla_front", shadow_size: int = 2048) -> None:
        self.size = size
        self.camera = camera
        self.shadow_size = shadow_size
        self._renderers: dict[int, mujoco.Renderer] = {}

    def __call__(self, sim: TennisSim) -> np.ndarray:
        r = self._renderers.get(id(sim))
        if r is None:
            sim.model.vis.quality.shadowsize = self.shadow_size
            r = mujoco.Renderer(sim.model, self.size, self.size)
            self._renderers[id(sim)] = r
        r.update_scene(sim.data, camera=self.camera)
        return r.render().copy()

    def close(self) -> None:
        for r in self._renderers.values():
            r.close()
        self._renderers.clear()
