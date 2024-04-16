import torch
import pytest
from utils.simulator import Simulator


@pytest.fixture
def simulator():
    return Simulator(n_steps=5)


def test_evaluate(simulator, OR_graph):
    inputs = torch.tensor([
        [0, 1, 0, 1],
        [0, 0, 1, 1],
    ], dtype=torch.bool)
    results = simulator.evaluate(OR_graph, inputs)
    assert torch.all(results == torch.tensor([
        [0, 1, 1, 1],
    ]))


def test_compare(simulator, OR_graph, NOR_graph):
    assert simulator.compare(OR_graph, NOR_graph) == 0.0
    assert simulator.compare(OR_graph, OR_graph) == 1.0


def test_satisfy_probability(simulator, FALSE_graph, TRUE_graph):
    assert simulator.satisfy_probability(FALSE_graph) == 0.0
    assert simulator.satisfy_probability(TRUE_graph) == 1.0
