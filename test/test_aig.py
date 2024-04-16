from utils.aig import AIGTranslator, ARITHS_GEN_MAP
from utils.cgp import CGPTranslator
from utils.simulator import Simulator
import torch


RAW_CGP = "{2,1,0,0,2,1,0}([2]0,1,3)([3]0,2,4)(3)"

def test_aig():
    cgp = CGPTranslator()
    aig = AIGTranslator()
    simulator = Simulator(n_steps=5)

    graph = cgp.parse(RAW_CGP)
    aig_graph = aig.translate(graph, ARITHS_GEN_MAP)

    inputs = torch.tensor([
        [0, 1, 0, 1],
        [0, 0, 1, 1],
    ], dtype=torch.bool)

    results = simulator.evaluate(aig_graph, inputs)
    assert torch.all(results == torch.tensor([
        [0, 0, 1, 0],
    ]))

