#!/usr/bin/env python3

import argparse
import os
from datetime import date
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description="Build a target in Universe")
    parser.add_argument(
        "-l", "--configs", nargs="+", help="List of config file paths to benchmark"
    )
    args = parser.parse_args()
    return args


def main():
    cur_date = str(date.today())
    args = parse_args()

    for config in args.configs:
        p = Path(config)
        run_name = f"benchmark_{p.stem}_{cur_date}"
        os.system(
            f"python3 ../benchmarks/bolt_benchmarks/bolt.py {config} --run_name {run_name}"
        )


if __name__ == "__main__":
    main()
