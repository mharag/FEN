import random

from torch.utils.data import Dataset
import torch
import os
from utils.cgp import CGPTranslator
from fen.model.fen import GraphData

class CGPDataset(Dataset):
    def __init__(self, dataset_dir, return_graphs=False):
        self.dir = dataset_dir
        self.files = os.listdir(self.dir)
        random.shuffle(self.files)
        self.files = self.files
        self.parser = CGPTranslator(include_constants=True)
        self.return_graphs = return_graphs

        if "metadata.json" in self.files:
            self.files.remove("metadata.json")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        file_name = self.files[idx]
        g1, g2, node_pairs, sim = self.load_file(os.path.join(self.dir, file_name))
        return g1, g2, node_pairs, sim

    def load_file(self, path):
        f = open(path, 'r')
        lines = f.readlines()
        f.close()
        c1, c2, *gt = lines
        node_pairs = []
        sim = []
        for line in gt:
            g1_node, g2_node, sim_raw = line.split(",")
            node_pairs.append([int(g1_node), int(g2_node)])
            sim.append(float(sim_raw))
        node_pairs = torch.tensor(node_pairs)
        sim = torch.tensor(sim)
        g1, g2 = self.parser.parse(c1), self.parser.parse(c2)
        if self.return_graphs:
            return g1, g2, float(sim)
        graph1 = GraphData.from_graph(g1)
        graph2 = GraphData.from_graph(g2)
        return graph1, graph2, node_pairs, sim
