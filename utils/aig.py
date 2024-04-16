import copy
from enum import Enum, auto
import torch
from utils.graph import Graph

# gate inputs
X_INPUT = -1
Y_INPUT = -2


class Gates(Enum):
    INPUT = auto()
    AND = auto()
    NOT = auto()
    OR = auto()
    XOR = auto()
    NAND = auto()
    NOR = auto()
    XNOR = auto()
    TRUE = auto()
    FALSE = auto()


# https://github.com/ehw-fit/ariths-gen/tree/main
ARITHS_GEN_MAP = {
    0: Gates.INPUT,
    1: Gates.NOT,
    2: Gates.AND,
    3: Gates.OR,
    4: Gates.XOR,
    5: Gates.NAND,
    6: Gates.NOR,
    7: Gates.XNOR,
    8: Gates.TRUE,
    9: Gates.FALSE,
}

AIG_MAP = {
    0: Gates.INPUT,
    1: Gates.AND,
    2: Gates.NOT
}

# "x", "y" represent original inputs
# int represents relative indexes to the position of original gate
GATE_TEMPLATES = {
    Gates.AND: {
        "nodes": [Gates.AND],
        "edges": [
            [X_INPUT, 0, 0],
            [Y_INPUT, 0, 1]
        ],
    },
    Gates.NOT: {
        "nodes": [Gates.NOT],
        "edges": [
            [X_INPUT, 0, 0],
            [Y_INPUT, 0, 0]
        ]
    },
    Gates.OR: {
        "nodes": [Gates.NOT, Gates.NOT, Gates.AND, Gates.NOT],
        "edges": [
            [X_INPUT, 0, 0],
            [X_INPUT, 0, 1],
            [Y_INPUT, 1, 0],
            [Y_INPUT, 1, 1],
            [0, 2, 0],
            [1, 2, 1],
            [2, 3, 0],
            [2, 3, 1],
        ]
    },
    Gates.XOR: {
        "nodes": [Gates.NOT, Gates.NOT, Gates.AND, Gates.AND, Gates.NOT, Gates.NOT, Gates.AND],
        "edges": [
            [X_INPUT, 0, 0],
            [X_INPUT, 0, 1],
            [Y_INPUT, 1, 0],
            [Y_INPUT, 1, 1],
            [0, 2, 0],
            [1, 2, 1],
            [X_INPUT, 3, 0],
            [Y_INPUT, 3, 1],
            [2, 4, 0],
            [2, 4, 1],
            [3, 5, 0],
            [3, 5, 1],
            [4, 6, 0],
            [5, 6, 1]
        ]
    },
    Gates.NAND: {
        "nodes": [Gates.AND, Gates.NOT],
        "edges": [
            [X_INPUT, 0, 0],
            [Y_INPUT, 0, 1],
            [0, 1, 0],
            [0, 1, 1]
        ]
    }
}


class AIGTranslator:
    """ Simple translator that translate circuits to AIG (And-Inverter Graph) format.

    Other gates like XOR, NAND... are replaced by predefined AIG templates.
    No optimization is performed.

    """
    def __init__(self):
        self.templates = self.preprocess_templates()

    def preprocess_templates(self):
        aig_map_inv = {v: k for k, v in AIG_MAP.items()}
        templates = {}
        for gate, template in GATE_TEMPLATES.items():
            nodes = torch.tensor([aig_map_inv[gate] for gate in template["nodes"]])
            edges = torch.tensor(template["edges"])
            templates[gate] = (nodes, edges)
        return templates

    def translate(self, g, gates_map):
        device = g.device

        gates_map_inv = {v: k for k, v in gates_map.items()}

        gate_sizes = torch.ones_like(g.nodes)
        for gate, (node_template, _) in self.templates.items():
            gate_sizes[g.nodes == gates_map_inv[gate]] = len(node_template)

        # relative shift is new size - original size
        rel_shift = gate_sizes - 1
        # how much did the end of the gate shift
        abs_end_shift = torch.cumsum(rel_shift, dim=0)
        # how much did start of the gate shift
        abs_start_shift = abs_end_shift - rel_shift

        # how much did the circuit size increase
        size_diff = abs_end_shift[-1]

        org_idx = torch.arange(g.n_nodes, device=device)
        new_idx = org_idx + abs_start_shift

        new_nodes = torch.zeros(int(g.n_nodes + size_diff), device=device, dtype=torch.int)
        new_edges = []

        for o, n in zip(org_idx, new_idx):
            # skip inputs
            if g.nodes[o].item() == 0:
                continue

            try:
                nodes_template, edges_template = self.templates[gates_map[g.nodes[o].item()]]
            except KeyError:
                raise ValueError(f"Gate {g.nodes[o]} not supported")

            old_x = g.edges[g.mask_edges_node_in(o) & g.mask_edges_x()][0][0]
            old_y = g.edges[g.mask_edges_node_in(o) & g.mask_edges_y()][0][0]
            new_x = old_x + abs_end_shift[old_x]
            new_y = old_y + abs_end_shift[old_y]
            new_nodes[n:n+gate_sizes[o]] = nodes_template

            gate_edges = copy.deepcopy(edges_template)
            # relative indices to absolute
            gate_edges[:, :2][gate_edges[:, :2] >= 0] = gate_edges[:, :2][gate_edges[:, :2] >= 0] + n
            # fill in input connections
            gate_edges[gate_edges == X_INPUT] = new_x
            gate_edges[gate_edges == Y_INPUT] = new_y

            new_edges.extend(gate_edges)

        new_edges = torch.stack(new_edges, dim=0)
        new_outputs = g.outputs + abs_end_shift[g.outputs]
        return Graph(
            new_nodes,
            new_edges,
            new_outputs
        )
