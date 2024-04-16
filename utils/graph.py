import torch
from env import device
import re
import copy

class Graph:
    def __init__(
        self,
        nodes,
        edges,
        outputs
    ):
        self.nodes = nodes
        self.edges = edges
        self.outputs = outputs

        self.n_nodes = len(nodes)
        self.n_inputs = torch.sum(nodes == 0)
        self.n_outputs = len(outputs)

        self.device = nodes.device

        self._adjacency_matrix = None
        self._forward_index = None

        self._mask_edges_x = None
        self._mask_edges_y = None

    @property
    def node_types(self):
        return torch.unique(self.nodes)

    def extract_output(self, values):
        return values[self.outputs]

    @property
    def adjacency_matrix(self):
        if self._adjacency_matrix is None:
            adj = torch.zeros((len(self.nodes), len(self.nodes)), device=device)
            for (parent, child, idx) in self.edges:
                adj[parent, child] += idx+1
            self._adjacency_matrix = adj
        return self._adjacency_matrix

    @property
    def forward_index(self):
        if self._forward_index is None:
            forward = torch.zeros_like(self.nodes, device=device) - 1
            forward[self.nodes == 0] = 0  # inputs

            last_unknown = torch.zeros_like(self.nodes, device=device).to(torch.bool)
            unknown = forward == -1
            while torch.max(unknown) and not torch.equal(unknown, last_unknown):
                adj = self.adjacency_matrix.clone()
                adj[unknown] = 0
                ready = torch.sum(adj, dim=0) == 3

                forward_2d = forward.repeat(len(self.nodes), 1).T
                forward_2d[self.adjacency_matrix == 0] = -1
                max_parent = torch.max(forward_2d, dim=0).values
                forward[ready] = max_parent[ready] + 1
                last_unknown = unknown
                unknown = forward == -1

            if torch.max(unknown):
                raise ValueError("Unable to calculate forward index.")
            self._forward_index = forward

        return self._forward_index

    def mask_adjacency_x(self):
        return (self.adjacency_matrix == 1) | (self.adjacency_matrix == 3)

    def mask_adjacency_y(self):
        return (self.adjacency_matrix == 2) | (self.adjacency_matrix == 3)

    def mask_edges_x(self):
        if self._mask_edges_x is None:
            self._mask_edges_x = self.edges[:, 2] == 0
        return self._mask_edges_x

    def mask_edges_y(self):
        if self._mask_edges_y is None:
            self._mask_edges_y = self.edges[:, 2] == 1
        return self._mask_edges_y

    def mask_edges_node_in(self, node_idx):
        return self.edges[:, 1] == node_idx

    def mask_edges_node_out(self, node_idx):
        return self.edges[:, 0] == node_idx

    def mask_nodes_input(self):
        return self.nodes == 0

    def mask_nodes_output(self):
        outputs = torch.zeros(self.n_nodes, device=device)
        outputs[self.outputs] = 1
        return outputs

    def subgraph(self, size):
        return Graph(
            self.nodes[:size],
            self.edges[self.edges[:, 1] < size],
            self.outputs[self.outputs < size]

        )
