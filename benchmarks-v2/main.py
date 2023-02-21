import argparse

from .runners.runner_map import runner_map
import re
from thirdai.experimental import MlflowCallback
from datetime import date
import thirdai


def parse_arguments():
    parser = argparse.ArgumentParser(description="Benchmark a dataset with Bolt")

    parser.add_argument(
        "--runner",
        type=str,
        required=True,
        choices=["udt", "bolt_fc", "dlrm"],
        help="Which runner to use to run the benchmark.",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="",
        help="Regular expression indicating which configs to run for the given runner.",
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
        "--run_name",
        type=str,
        default=None,
        help="The job name to track in MLflow",
    )
    parser.add_argument(
        "--official_benchmark",
        action="store_true",
        help="Controls if the experiment is logged to the '_benchmark' experiment or the regular experiment. This should only be used for the github actions benchmark runner.",
    )
    return parser.parse_args()


def experiment_name(config_name: str, official_benchmark: str):
    if official_benchmark:
        return f"{config_name}_benchmark"
    return config_name


if __name__ == "__main__":
    args = parse_arguments()

    runner = runner_map[args.runner.lower()]

    config_re = re.compile(args.config)
    configs = list(
        filter(
            lambda config: config_re.match(config.config_name),
            runner.config_type.__subclasses__(),
        )
    )
    if len(configs) == 0:
        raise ValueError(
            f"Could match regular expression '{args.config}' to any configs."
        )

    for config in configs:
        if args.mlflow_uri and args.run_name:
            mlflow_logger = MlflowCallback(
                tracking_uri=args.mlflow_uri,
                experiment_name=experiment_name(config.config_name, args.official_benchmark),
                run_name=f"{args.run_name}_{str(date.today())}",
                dataset_name=config.dataset_name,
                experiment_args={},
            )
            mlflow_logger.log_additional_param("thirdai_version", thirdai.__version__)
        else:
            mlflow_logger = None

        runner.run_benchmark(
            config=config,
            path=args.path_prefix,
            mlflow_logger=mlflow_logger
        )

        if mlflow_logger:
            mlflow_logger.end_run()
