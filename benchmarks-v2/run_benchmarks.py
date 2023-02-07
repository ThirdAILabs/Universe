#!/usr/bin/env python3

import os
import argparse
import subprocess
import json
import requests
from datetime import date
from pathlib import Path
from dotenv import load_dotenv
from mlflow_extraction import extract_mlflow_data


# This webhook is associated with the `MLFLOW Benchmarks` slack app. Currently, it posts
# messages to the #weekly_udt_benchmarks channel.
SLACK_WEBHOOK = (
    "https://hooks.slack.com/services/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"
)


def get_mlflow_uri():
    # load_dotenv() assumes that there is a file named .env
    # in the working directory, containing a variable `MLFLOW_URI`
    load_dotenv()
    return os.getenv("MLFLOW_URI")


def send_slack_message(experiment_name):
    df_md = extract_mlflow_data(experiment_name, markdown=True)
    payload = {"text": f"*{experiment_name}* \n ```{df_md}```"}
    return requests.post(SLACK_WEBHOOK, json.dumps(payload))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--test_run",
        action="store_true",
        default=False,
        help="Label a benchmarking job as a sanity test instead of an official run.",
    )
    parser.add_argument(
        "--engine",
        default="bolt",
        required=True,
        tehelp="Specify the engine(Bolt or UDT) to benchmark.",
    )
    return parser.parse_args()


def main():
    current_date = str(date.today())
    args = parse_args()

    prefix = "test_run" if args.test_run else "benchmark_run"

    engine = args.engine
    if engine.lower() == "bolt":
        from configs.bolt_configs import BoltBenchmarkConfig, DLRMConfig

        configs = BoltBenchmarkConfig.__subclasses__()
        configs.extend(DLRMConfig.__subclasses__())

    elif engine.lower() == "udt":
        from configs.udt_configs import UDTBenchmarkConfig

        configs = UDTBenchmarkConfig.__subclasses__()

    exit_code = 0
    mlflow_uri = get_mlflow_uri()
    for config in configs:
        config_name = config.__name__
        run_name = f"{prefix}_{current_date}"

        command = (
            f"python3 benchmarks-v2/benchmark_{engine.lower()}.py --mlflow_uri={mlflow_uri} "
            f"--run_name={run_name} --config_name={config_name}"
        )

        if (
            subprocess.call(
                command, shell=True, cwd=Path(__file__).resolve().parent.parent
            )
            != 0
        ):
            exit_code += 1

    if exit_code:
        exit(exit_code)

    for config in configs:
        send_slack_message(experiment_name=config.experiment_name)


if __name__ == "__main__":
    main()
