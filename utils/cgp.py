import re
import torch
from utils.graph import Graph

class CGPTranslator:
    """Class to parse and export CGP graphs.

    If implicit

    """
    def __init__(self, include_constants=False, device=None):
        self.include_constants = include_constants
        self.device = device

    def parse(self, raw_cgp):
        match = re.match(r"\{(.*)\}(.*)\(([\d,]*)\)", raw_cgp)
        head, body, tail = match.groups()
        c_in, c_out, c_rows, c_cols, c_ni, c_no, c_lback = map(int, head.split(","))
        nodes = body[1:-1].split(")(")
        nodes = [re.match(r"\[(\d*)\](\d*),(\d*),(\d*)", node).groups() for node in nodes]
        nodes = [list(map(int, node)) for node in nodes]
        tail = list(map(int, tail.split(",")))
        if self.include_constants:
            c_in += 2

        unsorted_nodes = []
        edges = []
        for i in range(c_in):
            unsorted_nodes.append([i, 0])
        for idx, top, bottom, func in nodes:
            unsorted_nodes.append([idx, func])
            edges.append([top, idx, 0])
            edges.append([bottom, idx, 1])

        nodes = torch.zeros(len(unsorted_nodes), dtype=torch.int)
        for i, func in unsorted_nodes:
            nodes[i] = func

        return Graph(nodes.to(self.device), torch.tensor(edges, device=self.device), torch.tensor(tail, device=self.device))

    def export(self, graph):
        c_in = graph.n_inputs - 2 if self.include_constants else graph.n_inputs
        x = torch.argmax(graph.mask_adjacency_x().to(torch.int), dim=0)
        y = torch.argmax(graph.mask_adjacency_y().to(torch.int), dim=0)
        head = f"{{{c_in},{len(graph.outputs)},0,0,2,1,0}}"
        body = [
            f"([{i}]{x[i]},{y[i]},{graph.nodes[i]})"
            for i in range(len(graph.nodes))
            if graph.nodes[i] != 0
        ]
        body = "".join(body)
        tail = "(" + ",".join(map(lambda x: str(x.item()), graph.outputs)) + ")"
        return head + body + tail
