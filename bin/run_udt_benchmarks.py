import argparse
import os
import subprocess
from datetime import date
from pathlib import Path
import inspect
import sys
import importlib.util

def get_udt_configs(universe_dir):
    udt_config_file = os.path.join(universe_dir, "benchmarks", "udt_configs.py")
    print(udt_config_file)
    spec = importlib.util.spec_from_file_location("udt_configs", udt_config_file)

    foo = importlib.util.module_from_spec(spec)

    sys.modules["udt_configs"] = foo
    spec.loader.exec_module(foo)
    clsmembers = inspect.getmembers(foo, inspect.isclass)

    return foo.UDTBenchmarkConfig.__subclasses__()

def parse_args():
    parser = argparse.ArgumentParser(description="Launch a benchmarking run")
    parser.add_argument(
        "-l",
        "--configs",
        nargs="+",
        required=False,
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
    universe_dir = Path(__file__).resolve().parent.parent
    prefix = "test_run" if args.test_run else "benchmark"
    # Exit code is the number of benchmarking tasks that failed
    exit_code = 0
    configs = get_udt_configs(universe_dir)
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
    exit(exit_code)


if __name__ == "__main__":
    main()
