import numpy as np
import pytest

from tennis.sim.config import TennisConfig
from tennis.sim.physics import TennisSim


@pytest.fixture(scope="session")
def cfg() -> TennisConfig:
    return TennisConfig()


@pytest.fixture()
def sim(cfg: TennisConfig) -> TennisSim:
    return TennisSim(cfg)


@pytest.fixture()
def rng() -> np.random.Generator:
    return np.random.default_rng(1234)
