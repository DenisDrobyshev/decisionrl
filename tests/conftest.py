"""Shared test fixtures."""

import pytest

from decisionrl.utils import Logger, set_seed


@pytest.fixture(autouse=True)
def _deterministic():
    """Seed every RNG before each test so results are order-independent.

    Network initialization uses the global PyTorch RNG, so without this a test's
    outcome could depend on which tests ran before it.
    """
    set_seed(0)


@pytest.fixture
def quiet_logger():
    return Logger(verbose=0)
