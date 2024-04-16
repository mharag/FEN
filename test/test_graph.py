import torch


def test_forward_index(OR_graph):
    assert OR_graph.forward_index == torch.tensor([0, 0, 1, 1, 2, 3])


def test_adjacency_matrix(OR_graph):
    assert torch.all(OR_graph.adjacency_matrix == torch.tensor([
        [0, 0, 3, 0, 0, 0],
        [0, 0, 0, 3, 0, 0],
        [0, 0, 0, 0, 1, 0],
        [0, 0, 0, 0, 2, 0],
        [0, 0, 0, 0, 0, 3],
        [0, 0, 0, 0, 0, 0]
    ]))

def test_subgraph(OR_graph):
    subgraph = OR_graph.subgraph(3)
    assert torch.all(subgraph.nodes == torch.tensor([0, 0, 2]))
    assert torch.all(subgraph.edges == torch.tensor([
        [0, 2, 0],
        [0, 2, 1],
    ]))
    assert torch.all(subgraph.outputs == torch.tensor([2]))
