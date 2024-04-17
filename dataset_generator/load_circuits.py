import os
from utils.graph import Graph
from utils.cgp import CGPTranslator


def load_circuits(files: list[str], include_constants: bool = False, device=None) -> list[Graph]:
    """Load circuits from a given path."""
    cgp_translator = CGPTranslator(include_constants=include_constants, device=device)

    circuits = []
    for file in files:
        f = open(file, "r")
        raw_cgp = f.readlines()[0]
        f.close()
        circuit = cgp_translator.parse(raw_cgp)
        circuit.name = file.rsplit(".", maxsplit=1)[0]
        circuits.append(circuit)

    return circuits
