#!/usr/bin/env python3

import argparse
import importlib.util
import inspect
import json
import os
import requests
import subprocess
from datetime import date
from pathlib import Path

from mlflow_extraction import extract_mlflow_data 

SLACK_WEBHOOK = "https://hooks.slack.com/services/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"

def parse_args():
    parser = argparse.ArgumentParser(description="Launch a benchmarking run")
    parser.add_argument(
        "--test_run",
        action="store_true",
        default=False,
        help="Flag to label the benchmarking job as a sanity test as opposed to an official run",
    )
    args = parser.parse_args()
    return args


def get_udt_configs(universe_dir):
    """
    This helper function loads the names of the UDT benchmark configs directly
    from the Universe/benchmarks/udt_configs.py file that defines these configs
    as python classes
    """
    udt_config_file = os.path.join(universe_dir, "benchmarks", "udt_configs.py")
    udt_config_spec = importlib.util.spec_from_file_location(
        "udt_configs", udt_config_file
    )
    udt_config_module = importlib.util.module_from_spec(udt_config_spec)
    udt_config_spec.loader.exec_module(udt_config_module)

    clsmembers = inspect.getmembers(udt_config_module, inspect.isclass)
    return udt_config_module.UDTBenchmarkConfig.__subclasses__()


def send_slack_message(experiment_name):
    df_md = extract_mlflow_data(experiment_name, markdown=True)
    payload = {"text": f"```{df_md}```"}
    return requests.post(SLACK_WEBHOOK, json.dumps(payload))


def main():
    cur_date = str(date.today())
    args = parse_args()
    universe_dir = Path(__file__).resolve().parent.parent
    prefix = "test_run" if args.test_run else "benchmark"
    # Exit code is the number of benchmarking tasks that failed
    exit_code = 0
    # configs = get_udt_configs(universe_dir)
    configs = []
    for config in configs:
        config = config.__name__
        run_name = f"{prefix}_{cur_date}"
        if (
            subprocess.call(
                f"python3 benchmarks/benchmark_udt.py --run_name {run_name} --config {config} ",
                shell=True,
                cwd=universe_dir,
            )
            != 0
        ):
            exit_code += 1
    
    if exit_code:
        exit(exit_code)
    else:
        configs = get_udt_configs(universe_dir)
        for config in configs:
            send_slack_message(config.experiment_name)


if __name__ == "__main__":
    main()
