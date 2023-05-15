import argparse
import os
from thirdai import bolt
from ...configs.cold_start_configs import *
from ...configs.graph_configs import *
from ...configs.mach_configs import *
from ...configs.udt_configs import *
from ...runners.runner_map import runner_map
from ...utils import get_configs

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
        "--model_folder",
        type=str,
        default="./old_models/",
        help="The path to the folder where old thirdai version models are saved",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()

    for runner_name in args.runner:
        runner = runner_map[runner_name.lower()]

        configs = get_configs(runner=runner, config_regex=args.config)

        for config in configs:
            model = runner.create_model(config, path_prefix="./benchmarks_v2/src/mini_benchmark_datasets/")

            if not os.path.exists(args.model_folder):
                # Create a new directory because it does not exist
                os.makedirs(args.model_folder)

            with open("thirdai.version") as version_file:
                version = version_file.read().strip()
                version = version.replace(".", "_")

            model.save(os.path.join(args.model_folder, f"{version}_{config.config_name}.model"))