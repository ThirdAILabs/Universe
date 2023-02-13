import argparse

from configs import bolt_configs, dlrm_configs, udt_configs
from runners import Runner


def parse_arguments():
    parser = argparse.ArgumentParser(description="Benchmark a dataset with Bolt")

    parser.add_argument(
        "--runner",
        required=True,
        help="Specify the runner name for the benchmark. Options include 'fully_connected', 'udt', 'dlrm'",
    )
    parser.add_argument(
        "--mlflow_uri", required=True, help="MLflow URI to log metrics and artifacts."
    )
    parser.add_argument(
        "--run_name", required=True, help="The job name to track in MLflow"
    )
    parser.add_argument(
        "--config_name",
        default="",
        required=True,
        help="The python class name of the benchmark config",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()

    if args.runner.lower() == "bolt":
        config = getattr(bolt_configs, args.config_name)

    elif args.runner.lower() == "udt":
        config = getattr(udt_configs, args.config_name)
    elif args.runner.lower() == "dlrm":
        config = getattr(dlrm_configs, args.config_name)

    else:
        raise ValueError(f"Invalid runner name: {args.runner}")

    runner = list(
        filter(
            lambda runner_class: runner_class.name == args.runner,
            Runner.__subclasses__(),
        )
    )
    runner = runner[0]
    runner.run_benchmark(
        config=config, mlflow_uri=args.mlflow_uri, run_name=args.run_name
    )
