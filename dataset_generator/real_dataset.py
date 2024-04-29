import argparse
import os
from dataset_generator.load_circuits import load_circuits
from utils.simulator import Simulator
import torch
from utils.aig import AIGTranslator, ARITHS_GEN_MAP
import sys
import json
from utils.cgp import CGPTranslator
import re
from progress.bar import Bar


def generate_pairs(circuits, size, samples, cross_circuit_pairs):
    pairs = []
    per_input_width = {}
    for c in circuits:
        if c.n_inputs not in per_input_width:
            per_input_width[c.n_inputs] = []
        per_input_width[c.n_inputs].append(c)

    bar = Bar("Generating pairs", max=samples)

    for i in range(samples):
        bar.next()
        c1 = circuits[torch.randint(0, len(circuits), (1,)).item()]
        if cross_circuit_pairs:
            # choose only circuits with the same input width
            pool = per_input_width[c1.n_inputs]
            c2 = pool[torch.randint(0, len(pool), (1,)).item()]
        else:
            c2 = c1

        # generate sub-graphs of random size from interval <size/2, size>
        s1 = c1.subgraph(size)
        s2 = c2.subgraph(size)

        # set output to the last node in graph
        s1.set_outputs(torch.tensor([s1.n_nodes-1]))
        s2.set_outputs(torch.tensor([s2.n_nodes-1]))

        pairs.append((s1, s2))
    bar.finish()
    return pairs


def save_dataset(samples, output, include_constants):
    os.makedirs(output)

    cgp = CGPTranslator(include_constants=include_constants)

    bar = Bar("Saving samples", max=len(samples))
    for i,samples in enumerate(samples):
        file_name = f"{i}.txt"
        path = os.path.join(output, file_name)
        g1, g2, similarities = samples
        f = open(path, "w")
        f.write(f"{cgp.export(g1)}\n")
        f.write(f"{cgp.export(g2)}\n")
        for sim in similarities:
            f.write(f"{sim[0]},{sim[1]},{sim[2]}\n")
        f.close()
        bar.next()
    bar.finish()


def add_metadata(samples, output, files, steps):
    path = os.path.join(output, "metadata.json")
    stats = {
        "_cmd": " ".join(sys.argv),
        "_source": files,
        "similarity": [],
        "graph_size": [],
        "input_width": [],
        "longest_path": [],
        "simulation_steps": steps,
    }
    for sample in samples:
        g1, g2, sim = sample
        stats["similarity"].append(sim)
        stats["graph_size"].append(g1.n_nodes)
        stats["graph_size"].append(g2.n_nodes)
        stats["input_width"].append(g2.n_inputs)
        stats["longest_path"].append(g1.forward_index[-1].item())
        stats["longest_path"].append(g2.forward_index[-1].item())

    with open(path, "w") as f:
        json.dump(stats, f, indent=4)


def augment_circuits(circuits, n_augmented, n_mutation):
    augmented_ciruits = []
    for c in circuits:
        augmented_ciruits.append(c)
        for i in range(n_augmented):
            mutated_circuit = c.mutate(n_mutation)
            augmented_ciruits.append(mutated_circuit)
    return augmented_ciruits


def create_dataset(
    files: list[str],
    size: int,
    samples: int,
    steps: int,
    output: str,
    include_constants: bool,
    cross_circuit_pairs: bool,
    n_per_pair: int,
    n_augmented: int,
    n_mutation: int,
    device: torch.device,
):
    print(f"Creating dataset {output}")
    if os.path.exists(output):
        raise ValueError(f"Output directory {output} already exists")

    circuits = load_circuits(files, include_constants)
    simulator = Simulator(steps, include_constants=include_constants, device=device)
    aig_translator = AIGTranslator()

    print(f"Number of circuits before augmentation: {len(circuits)}")
    circuits = augment_circuits(circuits, n_augmented, n_mutation)
    print(f"Number of circuits after augmentation: {len(circuits)}")

    pairs = generate_pairs(circuits, size, samples, cross_circuit_pairs)

    bar = Bar("Processing pairs", max=samples)
    samples = []
    for i, pair in enumerate(pairs):
        bar.next()
        g1, g2 = pair
        g1 = aig_translator.translate(g1, ARITHS_GEN_MAP).to(device)
        g2 = aig_translator.translate(g2, ARITHS_GEN_MAP).to(device)
        g1_nodes = torch.randint(g1.n_inputs, g1.n_nodes, (n_per_pair,), device=device)
        g2_nodes = torch.randint(g2.n_inputs, g2.n_nodes, (n_per_pair,), device=device)
        node_pairs = torch.stack((g1_nodes, g2_nodes), dim=-1)
        sim = simulator.compare(g1, g2, node_pairs)
        simmilarities = []
        for i in range(n_per_pair):
            simmilarities.append((g1_nodes[i].item(), g2_nodes[i].item(), sim[i].item()))
        samples.append((g1, g2, simmilarities))
    bar.finish()

    save_dataset(samples, output, include_constants)
    add_metadata(samples, output, files, steps)


def main():
    arg = argparse.ArgumentParser()
    arg.add_argument("--dataset", type=str)
    arg.add_argument("--size", type=int)
    arg.add_argument("--samples", type=int)
    arg.add_argument("--val_samples", type=int)
    arg.add_argument("--steps", type=int)
    arg.add_argument("--output", type=str)
    arg.add_argument("--include_constants", type=bool, default=False)
    arg.add_argument("--cross_circuit_pairs", type=bool, default=False)
    arg.add_argument("--val_filter", type=str, default=False)
    arg.add_argument("--use_cuda", type=bool, default=False)
    arg.add_argument("--n_per_pair", type=int)
    arg.add_argument("--n_augmented", type=int, default=0)
    arg.add_argument("--n_mutation", type=int, default=0)
    args = arg.parse_args()

    if not os.path.exists(args.dataset):
        raise ValueError(f"Dataset directory {args.dataset} does not exist")
    files = os.listdir(args.dataset)
    files = [os.path.join(args.dataset, f) for f in files]
    train_files, val_files = [], []
    for file in files:
        if re.search(args.val_filter, file):
            val_files.append(file)
        else:
            train_files.append(file)

    print(f"Files selected for training: {len(train_files)}")
    print(f"Files selected for validation: {len(val_files)}")

    device = torch.device("cuda" if args.use_cuda and torch.cuda.is_available() else "cpu")
    create_dataset(
        train_files,
        args.size,
        args.samples,
        args.steps,
        os.path.join(args.output, "train"),
        args.include_constants,
        args.cross_circuit_pairs,
        args.n_per_pair,
        args.n_augmented,
        args.n_mutation,
        device
    )
    if val_files:
        create_dataset(
            val_files,
            args.size,
            args.val_samples,
            args.steps,
            os.path.join(args.output, "val"),
            args.include_constants,
            args.cross_circuit_pairs,
            1,
            0,
                0,
            device
        )


if __name__ == '__main__':
    main()