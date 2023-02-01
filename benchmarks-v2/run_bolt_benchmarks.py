#!/usr/bin/env python3


import argparse
import subprocess
from datetime import date
from pathlib import Path

from configs.bolt_configs import BoltBenchmarkConfig


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--test_run",
        action="store_true",
        default=False,
        help="Label a benchmarking job as a sanity test instead of an official run.",
    )
    return parser.parse_args()


def main():
    current_date = str(date.today())
    args = parse_args()

    prefix = "test_run" if args.test_run else "benchmark_run"
    configs = BoltBenchmarkConfig.__subclasses__()

    exit_code = 0
    for config in configs:
        config_name = config.__name__
        run_name = f"{prefix}_{current_date}"

        command = f"python3 benchmarks-v2/runner.py --run_name={run_name} --config_name={config_name}"

        if (
            subprocess.call(
                command, shell=True, cwd=Path(__file__).resolve().parent.parent
            )
            != 0
        ):
            exit_code += 1

        exit()


if __name__ == "__main__":
    main()
