import numpy as np
import pytest

from hockey.observation import Geometry
from hockey.simulation import HockeySim, SimConfig


@pytest.fixture(scope="session")
def sim_cfg() -> SimConfig:
    return SimConfig()


@pytest.fixture()
def sim(sim_cfg: SimConfig) -> HockeySim:
    return HockeySim(sim_cfg)


@pytest.fixture(scope="session")
def geom(sim_cfg: SimConfig) -> Geometry:
    return Geometry.from_sim_config(sim_cfg)


@pytest.fixture()
def rng() -> np.random.Generator:
    return np.random.default_rng(1234)
