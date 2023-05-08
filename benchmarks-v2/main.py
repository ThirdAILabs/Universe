import argparse
import json
from datetime import date

import requests
import thirdai
from thirdai.experimental import MlflowCallback, MlflowCallbackV2

from .runners.runner_map import runner_map
from .runners.temporal import TemporalRunner
from .runners.udt import UDTRunner
from .utils import get_configs

# from runners.runner_map import runner_map
# from runners.temporal import TemporalRunner
# from runners.udt import UDTRunner
# from utils import get_configs


def parse_arguments():
    parser = argparse.ArgumentParser(description="Benchmark a dataset with Bolt")

    parser.add_argument(
        "--runner",
        type=str,
        nargs="+",
        required=True,
        choices=["udt", "bolt_fc", "dlrm", "query_reformulation", "temporal"],
        help="Which runners to use to run the benchmark.",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="",
        help="Regular expression indicating which configs to run for the given runners.",  # Empty string returns all configs for the given runners.
    )
    parser.add_argument(
        "--path_prefix",
        type=str,
        default="/share/data/",
        help="The path prefex to prepend to dataset paths. Defaults to '/share/data/'.",
    )
    parser.add_argument(
        "--mlflow_uri",
        type=str,
        default=None,
        help="MLflow URI to log metrics and artifacts.",
    )
    parser.add_argument(
        "--run_name", type=str, default=None, help="The job name to track in MLflow"
    )
    parser.add_argument(
        "--official_benchmark",
        action="store_true",
        help="Controls if the experiment is logged to the '_benchmark' experiment or the regular experiment. This should only be used for the github actions benchmark runner.",
    )
    parser.add_argument(
        "--slack_webhook",
        type=str,
        default="",
        help="Slack channel endpoint for posting messages to",
    )
    return parser.parse_args()


def experiment_name(config_name: str, official_benchmark: str):
    if official_benchmark:
        return f"{config_name}_benchmark"
    return config_name


if __name__ == "__main__":
    args = parse_arguments()

    for runner_name in args.runner:
        runner = runner_map[runner_name.lower()]

        configs = get_configs(runner=runner, config_regex=args.config)

        for config in configs:
            if args.mlflow_uri and args.run_name:
                mlflow_callback = MlflowCallback
                if runner == UDTRunner or runner == TemporalRunner:
                    mlflow_callback = MlflowCallbackV2

                mlflow_logger = mlflow_callback(
                    tracking_uri=args.mlflow_uri,
                    experiment_name=experiment_name(
                        config.config_name, args.official_benchmark
                    ),
                    run_name=f"{args.run_name}_{str(date.today())}",
                    experiment_args={"dataset": config.dataset_name},
                )
                mlflow_logger.log_additional_param(
                    "thirdai_version", thirdai.__version__
                )
            else:
                mlflow_logger = None

            try:
                runner.run_benchmark(
                    config=config,
                    path_prefix=args.path_prefix,
                    mlflow_logger=mlflow_logger,
                )
            except Exception as error:
                print(
                    f"An error occurred running the {config.config_name} benchmark:",
                    error,
                )
                if args.slack_webhook:
                    payload = f"{config.config_name} benchmark failed!"
                    requests.post(args.slack_webhook, json.dumps({"text": payload}))

            if mlflow_logger:
                mlflow_logger.end_run()
