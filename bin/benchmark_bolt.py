#!/usr/bin/env python3

import argparse
import os
from datetime import date
import subprocess
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
        "-m", "--model_type", default="engine", choices=["engine", "text_classifier"]
    )
    parser.add_argument(
        "--test_run",
        action="store_true",
        default=False,
        help="Flag to label the benchmarking job as a sanity test as opposed to an official run",
    )
    args = parser.parse_args()
    return args


def experiment_script(model_type):
    if model_type == "engine":
        return "bolt_benchmarks/run_bolt_experiment.py"
    if model_type == "text_classifier":
        return "text_classifier_benchmarks/run_text_classifier_experiment.py"
    else:
        raise ValueError(f"Model type cannot be {model_type}.")


def main():
    cur_date = str(date.today())
    args = parse_args()
    prefix = "test_run" if args.test_run else "benchmark"
    universe_dir = Path(__file__).resolve().parent.parent
    # Exit code is the number of benchmarking tasks that failed
    exit_code = 0
    for config in args.configs:
        p = Path(config)
        run_name = f"{prefix}_{p.stem}_{cur_date}"
<<<<<<< HEAD
        os.system(
            f"python3 {bin_directory}/../benchmarks/{experiment_script(args.model_type)} --disable_upload_artifacts --run_name {run_name}  {config} "
        )
=======
        os.path.dirname(os.path.realpath(__file__))
        if (
            subprocess.call(
                f"python3 benchmarks/bolt.py --disable_upload_artifacts --run_name {run_name}  {config} ",
                shell=True,
                cwd=universe_dir,
            )
            != 0
        ):
            exit_code += 1
    exit(exit_code)
>>>>>>> 56f2b447317f6447c102498eb69c1187140b7e50


if __name__ == "__main__":
    main()
