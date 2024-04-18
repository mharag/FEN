from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import os
from torch import nn
from fen.model.mlp import MLP
from torch_geometric.data import Data


class GraphData(Data):
    def __init__(self, nodes, x_edges, y_edges, outputs):
        super().__init__()
        self.nodes = nodes
        self.x_edges = x_edges
        self.y_edges = y_edges
        self.outputs = outputs

    @classmethod
    def from_graph(cls, graph):
        x_edges = torch.cat([
            torch.zeros(graph.n_inputs, dtype=torch.int),
            graph.edges[graph.edges[:, 2] == 0][:, 0]
        ])
        y_edges = torch.cat([
            torch.zeros(graph.n_inputs, dtype=torch.int),
            graph.edges[graph.edges[:, 2] == 1][:, 0]
        ])
        return cls(
            nodes=graph.nodes,
            x_edges=x_edges,
            y_edges=y_edges,
            outputs=graph.outputs
        )


class FEN(nn.Module):
    def __init__(
        self,
        n_embd = 256,
        n_layers = 10,
        n_inputs = 66,
    ):
        super(FEN, self).__init__()

        self.n_embd = n_embd
        self.mlp = MLP(
            n_in=2 * n_embd,
            n_hidden=n_embd * 4,
            n_out=n_embd,
            n_layers=n_layers,
            p_dropout=0.05,
        )

        self.input_embd = nn.Embedding(n_inputs, self.n_embd)
        self.normalization = nn.LayerNorm(self.n_embd)

        self._init_weights()

    def _init_weights(self):
        self.input_embd.weight.data.uniform_(-1, 1)

    @property
    def device(self):
        return next(self.parameters()).device

    def n_param(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def forward(self, g: GraphData):
        n_inputs = torch.sum(g.nodes == 0).item()
        node_embd = torch.cat([
            self.input_embd(torch.arange(n_inputs, device=self.device)),
            torch.zeros(g.nodes.size(0) - n_inputs, self.n_embd, device=self.device)
        ])

        and_mask = g.nodes == 1
        not_mask = g.nodes == 2

        x_ready = torch.zeros(g.nodes.size(0), dtype=torch.bool, device=self.device)
        y_ready = torch.zeros(g.nodes.size(0), dtype=torch.bool, device=self.device)
        done = torch.zeros(g.nodes.size(0), dtype=torch.bool, device=self.device)

        ready = (g.nodes == 0) | (x_ready & y_ready)
        while torch.any(ready):
            node_embd[ready & not_mask] = -node_embd[g.x_edges[ready & not_mask]]
            node_embd[ready & and_mask] = self.normalization(self.mlp(
                torch.cat([
                    node_embd[g.x_edges[ready & and_mask]],
                    node_embd[g.y_edges[ready & and_mask]],
                ], dim=-1)
            ))
            processed = ready.nonzero()
            x_ready[torch.isin(g.x_edges, processed)] = True
            y_ready[torch.isin(g.y_edges, processed)] = True
            done[ready] = True

            ready = x_ready & y_ready & ~done

        return node_embd

    def load(self, path):
        self.load_state_dict(torch.load(path)["state_dict"])

    def save(self, path):
        torch.save({"state_dict": self.state_dict()}, path)
