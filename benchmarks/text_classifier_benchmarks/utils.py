import mlflow
import os
import toml
import platform
import psutil
import socket
import sys
import numpy as np

from typing import Any, Dict
from sklearn.datasets import load_svmlight_file


def start_mlflow(experiment_name, run_name, dataset, model_name):
    file_dir = os.path.dirname(os.path.abspath(__file__))
    file_name = os.path.join(file_dir, "../config.toml")
    with open(file_name) as f:
        parsed_config = toml.load(f)
        mlflow.set_tracking_uri(parsed_config["tracking"]["uri"])

    mlflow.set_experiment(experiment_name)
    mlflow.start_run(
        run_name=run_name,
        tags={"dataset": dataset, "model": model_name},
    )


def log_machine_info():
    machine_info = {
        "load_before_experiment": os.getloadavg()[2],
        "platform": platform.platform(),
        "platform_version": platform.version(),
        "platform_release": platform.release(),
        "architecture": platform.machine(),
        "processor": platform.processor(),
        "hostname": socket.gethostname(),
        "ram_gb": round(psutil.virtual_memory().total / (1024.0**3)),
        "num_cores": psutil.cpu_count(logical=True),
    }

    mlflow.log_params(machine_info)


def config_get(config, field):
    if field not in config:
        raise ValueError(
            f'The field "{field}" was expected to be in "{config}" but was not found.'
        )
    return config[field]


def config_get_or(config, field, default):
    if field in config:
        return config[field]
    return default

def find_full_filepath(filename: str) -> str:
    if os.path.exists(filename):
        return filename

    # Load path prefixes to look for datasets from config file in the repository.
    data_path_file = (
        os.path.dirname(os.path.abspath(__file__)) + "/../../dataset_paths.toml"
    )

    prefix_table = toml.load(data_path_file)

    # Collect prefixes from environment variable. Configured similiar to *nix
    # or Windows PATH variable, separated by path sep.
    env_prefix_paths = os.environ.get("THIRDAI_DATASET_PATH", "").split(os.pathsep)

    # Configure environment variable prefix paths to have more priority over
    # config prefix paths.
    prefixes = env_prefix_paths + prefix_table["prefixes"]

    for prefix in prefixes:
        candidate_path = os.path.join(prefix, filename)
        if os.path.exists(candidate_path):
            return candidate_path
    print(
        "Could not find file '"
        + filename
        + "' on any filepaths. Add correct path to 'Universe/dataset_paths.toml or specify in THIRDAI_DATASET_PATH environment variable'"
    )
    sys.exit(1)
