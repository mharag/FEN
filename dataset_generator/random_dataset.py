# needs refactor


from graph import Graph
import torch
import random
from env import device



def random_graph(n_inputs, n_outputs, n_nodes, l_back=10):
    nodes = torch.randint(1, 3, (n_nodes+n_inputs,), device=device)
    nodes[:n_inputs] = 0
    connections = [
        (
            i,
            random.randint(max(0, i-1-l_back), i-1),
            random.randint(max(0, i-1-l_back), i-1)
        )
        for i in range(n_inputs, n_nodes+n_inputs)
    ]
    edges = [
        [bottom if idx else top, child, idx]
        for idx in [0, 1] for child, top, bottom in connections
    ]
    edges = torch.tensor(edges, device=device)
    outputs = torch.arange(n_inputs+n_nodes-n_outputs, n_inputs+n_nodes, 1, device=device)
    return Graph(nodes, edges, outputs)


def maximize_information(graph, sim=100, temperature=1.0, l_back=10):
    values = torch.zeros(graph.n_nodes, sim, device=device)
    for i in range(graph.n_inputs):
        values[i] = torch.randint(0, 2, (sim,), device=device)

    one_probability = torch.sum(values, dim=1) / float(sim)
    zero_probability = 1 - one_probability
    max_probability = torch.max(one_probability, zero_probability)
    max_probability[max_probability > 0.999] = -10
    select_probability = torch.exp(max_probability / temperature)

    for i in range(graph.n_inputs, graph.n_nodes):
        visible_ancestors = max(0, i-l_back)
        top = torch.multinomial(select_probability[visible_ancestors:i], 1).item() + visible_ancestors
        bot = torch.multinomial(select_probability[visible_ancestors:i], 1).item() + visible_ancestors
        node_edges = graph.edges[:, 1] == i
        top_edges = graph.edges[:, 2] == 0

        graph.edges[node_edges & top_edges, 0] = top
        graph.edges[node_edges & ~top_edges, 0] = bot

        graph.forward_index[i] = max(graph.forward_index[top], graph.forward_index[bot]) + 1
        graph.adjacency_matrix[:, i] = 0
        graph.adjacency_matrix[top, i] += 1
        graph.adjacency_matrix[bot, i] += 2

        value = torch.zeros(sim, device=device)
        if graph.nodes[i] == 1:
            value[(values[top] + values[bot] == 2)] = 1
        elif graph.nodes[i] == 2:
            value[(values[top] == 0)] = 1
        values[i] = value
        one_prob = torch.sum(value) / float(sim)
        zero_prob = 1 - one_prob
        max_prob = max(one_prob, zero_prob)
        max_prob = 0 if max_prob == 1 else max_prob
        select_prob = torch.exp(torch.tensor([max_prob]) / temperature)
        one_probability[i] = one_prob
        select_probability[i] = select_prob

    return graph