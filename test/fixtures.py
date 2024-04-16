import pytest
import torch
from utils.graph import Graph

@pytest.fixture
def OR_graph():
    # OR gate
    nodes = torch.tensor([2, 2, 1, 2])
    edges = torch.tensor([
        [0, 2, 0],
        [0, 2, 1],
        [1, 3, 0],
        [1, 3, 1],
        [2, 4, 0],
        [3, 4, 1],
        [4, 5, 0],
        [4, 5, 1]
    ])
    return Graph(nodes, edges, torch.tensor([5]))


@pytest.fixture
def NOR_graph():
    # OR gate
    nodes = torch.tensor([2, 2, 1])
    edges = torch.tensor([
        [0, 2, 0],
        [0, 2, 1],
        [1, 3, 0],
        [1, 3, 1],
        [2, 4, 0],
        [3, 4, 1],
    ])
    return Graph(nodes, edges, torch.tensor([4]))