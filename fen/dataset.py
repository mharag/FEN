import random

from torch.utils.data import Dataset
import os
from utils.cgp import CGPTranslator
from fen.model.fen import GraphData

class CGPDataset(Dataset):
    def __init__(self, dataset_dir):
        self.dir = dataset_dir
        self.files = os.listdir(self.dir)
        random.shuffle(self.files)
        self.files = self.files
        self.parser = CGPTranslator(include_constants=True)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        file_name = self.files[idx]
        g1, g2, sim = self.load_file(os.path.join(self.dir, file_name))
        return g1, g2, sim

    def load_file(self, path):
        f = open(path, 'r')
        lines = f.readlines()
        f.close()
        c1, c2, sim = lines
        graph1 = GraphData.from_graph(self.parser.parse(c1))
        graph2 = GraphData.from_graph(self.parser.parse(c2))
        return graph1, graph2, float(sim)