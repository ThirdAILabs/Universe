#!/usr/bin/env python3

import argparse
import json
import os
import subprocess
from datetime import date
from pathlib import Path

import requests
import toml
from dotenv import load_dotenv
from mlflow_extraction import extract_mlflow_data

# This webhook is associated with the `MLFLOW Benchmarks` slack app. Currently, it posts
# messages to the #weekly_udt_benchmarks channel.
SLACK_WEBHOOK = (
    "https://hooks.slack.com/services/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"
)


def get_mlflow_uri():
    file_dir = os.path.dirname(os.path.abspath(__file__))
    file_name = os.path.join(file_dir, "config.toml")
    with open(file_name) as f:
        parsed_config = toml.load(f)
        return parsed_config["tracking"]["uri"]


def send_slack_message(mlflow_uri, experiment_name):
    df_md = extract_mlflow_data(mlflow_uri=mlflow_uri, experiment_name=experiment_name, markdown=True)
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
        "--runner",
        required=True,
        help="Specify the runner type for the current benchmark. "
        "For Bolt options include 'fully_connected' and 'dlrm'. "
        "For UDT, the runner should also be 'udt'",
    )

    return parser.parse_args()


def main():
    current_date = str(date.today())
    args = parse_args()

    prefix = "test_run" if args.test_run else "benchmark_run"

    runner = args.runner

    if runner.lower() == "fully_connected":
        from configs.bolt_configs import BoltBenchmarkConfig

        configs = BoltBenchmarkConfig.__subclasses__()

    elif runner.lower() == "dlrm":
        from configs.dlrm_configs import DLRMConfig

        configs = DLRMConfig.__subclasses__()

    elif runner.lower() == "udt":
        from configs.udt_configs import UDTBenchmarkConfig

        configs = UDTBenchmarkConfig.__subclasses__()

    else:
        raise ValueError(f"Invalid benchmark runner: {runner}")

    exit_code = 0
    mlflow_uri = get_mlflow_uri()
    for config in configs:
        config_name = config.__name__
        if config_name != "AmazonPolarityConfig":
            continue
        run_name = f"{prefix}_{current_date}"

        command = (
            f"python3 benchmarks-v2/main.py "
            f"--runner={runner} "
            f"--mlflow_uri={mlflow_uri} "
            f"--run_name={run_name} "
            f"--config_name={config_name}"
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
        send_slack_message(mlflow_uri=mlflow_uri, experiment_name=config.experiment_name)


if __name__ == "__main__":
    main()
