import torch

class Simulator:
    def __init__(self, n_steps, device=None, include_constants=False, dtype=torch.bool):
        self.n_steps = n_steps
        self.device = device
        self.include_constants = include_constants
        self.dtype = dtype

    @staticmethod
    def _assert_aig(graph):
        used_gates = graph.node_types
        assert torch.max(used_gates) <= 2, "Only AIGs are supported"

    def _generate_random_inputs(self, n_inputs):
        if self.include_constants:
            n_inputs = n_inputs - 2
        inputs = torch.randint(0, 2, (n_inputs, self.n_steps), dtype=self.dtype, device=self.device)
        return inputs

    def evaluate(self, g, inputs, output_inner_nodes=False):
        self._assert_aig(g)

        # to support single batch of inputs
        if len(inputs.shape) == 1:
            inputs = inputs.unsqueeze(0)

        n_batch = inputs.shape[1]

        if self.include_constants:
            inputs = torch.cat((
                torch.zeros(1, n_batch, device=self.device, dtype=self.dtype),
                torch.ones(1, n_batch, device=self.device, dtype=self.dtype),
                inputs
            ), dim=0)

        values = torch.zeros(g.n_nodes, n_batch, device=self.device, dtype=self.dtype)
        values[g.mask_nodes_input()] = inputs

        for i in range(g.n_inputs, g.n_nodes):
            x = g.edges[g.mask_edges_node_in(i) & g.mask_edges_x()][0][0]
            y = g.edges[g.mask_edges_node_in(i) & g.mask_edges_y()][0][0]
            if g.nodes[i] == 1:  # AND
                values[i][values[x] & values[y]] = 1
            elif g.nodes[i] == 2:  # NOT
                values[i][~values[x]] = 1

        if output_inner_nodes:
            return values
        else:
            return values[g.mask_nodes_output()]

    def compare(self, g1, g2, node_pairs=None):
        assert g1.n_inputs == g2.n_inputs, "Graphs must have the same number of inputs"
        inputs = self._generate_random_inputs(g1.n_inputs)

        v1 = self.evaluate(g1, inputs, output_inner_nodes=True)
        v2 = self.evaluate(g2, inputs, output_inner_nodes=True)
        return torch.sum(v1[node_pairs[:, 0]] == v2[node_pairs[:, 1]], dim=-1) / float(self.n_steps)

    def satisfy_probability(self, graph):
        inputs = self._generate_random_inputs(graph.n_inputs)
        values = self.evaluate(graph, inputs)
        return torch.sum(values, dim=1) / float(self.n_steps)
