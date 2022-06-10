import mlflow
import os
import toml
import platform
import psutil
import socket

from typing import Any, Dict


def start_mlflow(experiment_name, run_name, dataset):
    file_dir = os.path.dirname(os.path.abspath(__file__))
    file_name = os.path.join(file_dir, "../config.toml")
    with open(file_name) as f:
        parsed_config = toml.load(f)
        mlflow.set_tracking_uri(parsed_config["tracking"]["uri"])

    mlflow.set_experiment(experiment_name)
    mlflow.start_run(
        run_name=run_name,
        tags={"dataset": dataset},
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


def log_subconfig(name: str, subconfig: Dict[str, Any]):
    for param, val in subconfig.items():
        mlflow.log_param(f"{name}_{param}", val)


def log_config_info(config):
    for name, subconfig in config.items():
        if isinstance(subconfig, dict):
            log_subconfig(name, subconfig)
        if isinstance(subconfig, list):
            for i, subconfig_i in enumerate(subconfig):
                log_subconfig(f"{name}_{i}", subconfig_i)
