import argparse
import json
from datetime import date
from types import SimpleNamespace

import requests
import thirdai
from thirdai.experimental import MlflowCallback

from .runners.runner_map import runner_map
from .runners.temporal import TemporalRunner
from .runners.udt import UDTRunner
from .utils import get_configs


def parse_arguments():
    parser = argparse.ArgumentParser(description="Benchmark a dataset with Bolt")

    parser.add_argument(
        "--runner",
        type=str,
        nargs="+",
        required=True,
        choices=list(runner_map.keys()),
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
        help="MLflow URI to log metrics and artifacts.",
    )
    parser.add_argument(
        "--run_name", type=str, default=None, help="The job name to track in MLflow"
    )

    # Do not set the following args when running this script manually, these are only for github actions
    parser.add_argument(
        "--official_slack_webhook",
        type=str,
        default="",
        help="Slack channel endpoint for official benchmarks",
    )
    parser.add_argument(
        "--branch_slack_webhook",
        type=str,
        default="",
        help="Slack channel endpoint for branch benchmarks",
    )
    parser.add_argument(
        "--branch_name",
        type=str,
        default="",
        help="Name of branch that benchmarks are being run on",
    )
    parser.add_argument(
        "--config_type",
        type=str,
        default=None,
        help="If specified, will ensure that each config to be run has a config_type field equal to this value.",
    )
    return parser.parse_args()


def experiment_name(config_name, official_benchmark):
    if official_benchmark:
        return f"{config_name}_benchmark"
    return config_name


def main(**kwargs):
    if not kwargs:
        args = parse_arguments()
    else:
        args = SimpleNamespace(**kwargs)

    if args.branch_name == "main":
        slack_webhook = args.official_slack_webhook
    elif args.branch_name != "":
        slack_webhook = args.branch_slack_webhook
    else:
        slack_webhook = ""

    # If benchmarks are called from github action, run_name = branch_name
    if args.branch_name:
        args.run_name = args.branch_name

    # If any of the benchmarks fail, we throw an error at the end of the script
    throw_exception = False

    for runner_name in args.runner:
        runner = runner_map[runner_name.lower()]

        configs = get_configs(
            runner=runner,
            config_regex=args.config,
            config_type=args.config_type if hasattr(args, "config_type") else None,
        )

        for config in configs:
            if args.mlflow_uri and args.run_name:
                mlflow_logger = MlflowCallback(
                    tracking_uri=args.mlflow_uri,
                    experiment_name=experiment_name(
                        config.config_name,
                        official_benchmark=args.branch_name == "main",
                    ),
                    run_name=f"{args.run_name}_{str(date.today())}",
                    experiment_args={"dataset": config.dataset_name},
                )
                mlflow_logger.log_additional_param(
                    "thirdai_version", thirdai.__version__
                )
                mlflow_logger.log_additional_param("runner", runner_name)
            else:
                mlflow_logger = None

            try:
                runner.run_benchmark(
                    config=config,
                    path_prefix=args.path_prefix,
                    mlflow_logger=mlflow_logger,
                )
            except Exception as error:
                throw_exception = True
                print(
                    f"An error occurred running the {config.config_name} benchmark:",
                    error,
                )
                payload = f"{config.config_name} benchmark failed on the {args.branch_name} branch!"
                if slack_webhook:
                    requests.post(slack_webhook, json.dumps({"text": payload}))
                else:
                    print(payload)

            if mlflow_logger:
                mlflow_logger.end_run()

    if throw_exception:
        raise Exception("One or more benchmark runs have failed")


if __name__ == "__main__":
    main()
