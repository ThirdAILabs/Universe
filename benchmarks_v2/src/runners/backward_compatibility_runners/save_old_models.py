import argparse
import os

from thirdai import bolt

from ...configs.cold_start_configs import *
from ...configs.graph_configs import *
from ...configs.mach_configs import *
from ...configs.udt_configs import *
from ...runners.runner_map import runner_map
from ...utils import get_configs
from .utils import OLD_MODEL_PATH


def parse_arguments():
    parser = argparse.ArgumentParser(description="Benchmark a dataset with Bolt")
    parser.add_argument(
        "--runner",
        type=str,
        nargs="+",
        required=True,
        choices=["udt", "query_reformulation", "temporal"],
        help="The runner to retrieve configs for.",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="",
        help="Regular expression indicating which configs to retrieve for the given runners.",  # Empty string returns all configs for the given runners.
    )
    parser.add_argument(
        "--version",
        type=str,
        default="",
        help="thirdai version that is being used to save the model",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()

    for runner_name in args.runner:
        runner = runner_map[runner_name.lower()]

        configs = get_configs(runner=runner, config_regex=args.config)

        for config in configs:
            model = runner.create_model(
                config, path_prefix="./benchmarks_v2/src/mini_benchmark_datasets/"
            )

            if not os.path.exists(OLD_MODEL_PATH):
                os.makedirs(OLD_MODEL_PATH)

            version = args.version.replace(".", "_")
            model.save(
                os.path.join(OLD_MODEL_PATH, f"{version}_{config.config_name}.model")
            )
