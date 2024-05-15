import argparse
import os

import numpy as np
import pandas as pd
from thirdai import bolt

from ..runners.runner_map import runner_map
from ..utils import get_configs


def parse_arguments():
    parser = argparse.ArgumentParser(description="Benchmark a dataset with Bolt")
    parser.add_argument(
        "--runner",
        type=str,
        nargs="+",
        required=True,
        choices=["udt", "bolt_fc", "dlrm", "query_reformulation", "temporal"],
        help="The runner to retrieve configs for.",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="",
        help="Regular expression indicating which configs to retrieve for the given runners.",  # Empty string returns all configs for the given runners.
    )
    parser.add_argument(
        "--read_path_prefix",
        type=str,
        default="/share/data/",
        help="The path prefix to read the original benchmark datasets from",
    )
    parser.add_argument(
        "--write_path_prefix",
        type=str,
        default="./benchmarks/mini_benchmark_datasets/",
        help="The path prefix to write the mini benchmark datasets to",
    )
    return parser.parse_args()


def save_non_graph_subset(file, num_lines=11):
    if not file:
        return

    read_path = os.path.join(args.read_path_prefix, file)

    with open(read_path) as input_file:
        head = [next(input_file) for _ in range(num_lines)]

    write_path = os.path.join(args.write_path_prefix, file)
    if not os.path.exists(os.path.dirname(write_path)):
        os.makedirs(os.path.dirname(write_path))

    with open(write_path, "w") as output_file:
        output_file.writelines(head)


def save_data_subset(config, num_lines=11):
    try:
        has_gnn_backend = any(
            [
                type(t) == bolt.types.neighbors
                for t in config.get_data_types(args.read_path_prefix).values()
            ]
        )
    except:
        has_gnn_backend = False

    if not has_gnn_backend:
        if hasattr(config, "train_file"):
            save_non_graph_subset(config.train_file, num_lines)

        if hasattr(config, "test_file"):
            save_non_graph_subset(config.test_file, num_lines)

        if hasattr(config, "cold_start_train_file"):
            save_non_graph_subset(config.cold_start_train_file, num_lines)

    else:
        # We create a small dummy graph dataset from a given file graph csv files
        # Since we only use a small subset of nodes from the graph, we must recreate the neighbors of the nodes
        # to form a valid graph.

        data_types = config.get_data_types(args.read_path_prefix)

        # Get column name corresponding to node_id type from data_types
        node_id_col = [
            k for k, v in data_types.items() if isinstance(v, bolt.types.node_id)
        ][0]

        # Get column name corresponding to neighbors type from data_types
        neighbors_col = [
            k for k, v in data_types.items() if isinstance(v, bolt.types.neighbors)
        ][0]

        read_test_file_path = os.path.join(args.read_path_prefix, config.test_file)

        write_train_file_path = os.path.join(args.write_path_prefix, config.train_file)
        write_test_file_path = os.path.join(args.write_path_prefix, config.test_file)
        write_gnn_index_path = os.path.join(
            args.write_path_prefix, os.path.dirname(config.test_file), "gnn_index.csv"
        )
        if not os.path.exists(os.path.dirname(write_train_file_path)):
            os.makedirs(os.path.dirname(write_train_file_path))

        test_df = pd.read_csv(read_test_file_path, nrows=num_lines - 1)

        test_df[config.target] = np.random.randint(
            config.n_target_classes, size=num_lines - 1
        )

        # Re-index the nodes in the graph
        test_df[node_id_col] = list(range(num_lines - 1))

        # Create valid neighbors
        neighbor_matrix = np.random.randint(0, 2, size=(num_lines - 1, num_lines - 1))
        # Create symmetric matrix for undirected graph by taking ths lower triangular matrix (tril)
        # and setting it to the upper triangular matrix as well (tril.T)
        neighbor_matrix = np.tril(neighbor_matrix) + np.tril(neighbor_matrix, -1).T
        neighbors = [
            " ".join([str(nb) for nb in np.nonzero(nbs)[0]]) for nbs in neighbor_matrix
        ]
        test_df[neighbors_col] = neighbors

        test_df.to_csv(write_test_file_path)

        train_df = test_df.sample(frac=0.5)
        train_df.to_csv(write_train_file_path)

        # The gnn index is the entire graph without the ground truth targets (targets set to 0)
        test_df[config.target] = 0
        test_df.to_csv(write_gnn_index_path)


if __name__ == "__main__":
    args = parse_arguments()

    for runner_name in args.runner:
        runner = runner_map[runner_name.lower()]

        configs = get_configs(runner=runner, config_regex=args.config)

        for config in configs:
            save_data_subset(config)
