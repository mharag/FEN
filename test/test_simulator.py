import torch
import pytest
from utils.simulator import Simulator


@pytest.fixture
def simulator():
    return Simulator(n_steps=5)


def test_evaluate(simulator, OR_graph):
    simulator.evaluate(OR_graph)