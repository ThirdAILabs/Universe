#!/usr/bin/env python3

import argparse
import os
from datetime import date
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description="Launch a benchmarking run")
    parser.add_argument(
        "-l",
        "--configs",
        nargs="+",
        required=True,
        help="List of config file paths to benchmark",
    )
    parser.add_argument(
        "--test_run",
        action="store_true",
        default=False,
        help="Flag to label the benchmarking job as a sanity test as opposed to an official run",
    )
    args = parser.parse_args()
    return args


def main():
    cur_date = str(date.today())
    args = parse_args()
    prefix = "test_run" if args.test_run else "benchmark"
    bin_directory = os.path.dirname(os.path.abspath(__file__))
    for config in args.configs:
        p = Path(config)
        run_name = f"{prefix}_{p.stem}_{cur_date}"
        os.system(
            f"python3 {bin_directory}/../benchmarks/bolt_benchmarks/bolt.py {config} --disable_upload_artifacts --run_name {run_name}"
        )


if __name__ == "__main__":
    main()
