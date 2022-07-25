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


def start_mlflow(config, mlflow_args):
    if not mlflow_args.disable_mlflow:
        experiment_name = config["experiment_identifier"]
        dataset_name = config["dataset_identifier"]
        model_name = config["model_identifier"]
        start_mlflow_helper(
            experiment_name, mlflow_args.run_name, dataset_name, model_name
        )
        log_machine_info()
        if mlflow_args.upload_artifacts:
            mlflow.log_artifact(config)


def start_mlflow_helper(experiment_name, run_name, dataset, model_name):
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


def log_single_epoch_training_metrics(train_output):
    # Since train_output is the result of training a single epoch,
    # we can greatly simplify the logging:
    mlflow_metrics = {k: v[0] for k, v in train_output.items()}
    mlflow.log_metrics(mlflow_metrics)


def log_prediction_metrics(inference_output):
    # The metrics data is the first element of the inference output tuple
    mlflow.log_metrics(inference_output[0])


def verify_mlflow_args(parser, mlflow_args):
    if not mlflow_args.disable_mlflow and not mlflow_args.run_name:
        parser.print_usage()
        raise ValueError("Error: --run_name is required when using mlflow logging.")


def config_get(config, field):
    if field not in config:
        raise ValueError(
            f'The field "{field}" was expected to be in "{config}" but was not found.'
        )
    return config[field]


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


def find_full_filepath(filename: str) -> str:
    data_path_file = (
        os.path.dirname(os.path.abspath(__file__)) + "/../../dataset_paths.toml"
    )
    prefix_table = toml.load(data_path_file)
    for prefix in prefix_table["prefixes"]:
        if os.path.exists(prefix + filename):
            return prefix + filename
    print(
        "Could not find file '"
        + filename
        + "' on any filepaths. Add correct path to 'Universe/dataset_paths.toml'"
    )
    sys.exit(1)


def _list_of_lists_to_csr(lists, use_softmax):
    offsets = np.zeros(shape=(len(lists) + 1,), dtype="uint32")
    for i in range(1, len(offsets)):
        offsets[i] = offsets[i - 1] + len(lists[i - 1])
    values = np.ones(shape=(offsets[-1],)).astype("float32")
    if use_softmax:
        for i in range(1, len(offsets)):
            start = offsets[i - 1]
            end = offsets[i]
            length = end - start
            for j in range(start, end):
                values[j] /= length
    indices = np.concatenate(lists).astype("uint32")
    return (indices, values, offsets)


# CSR format is the format we typically use to represent sparse matrices,
# (indices, values, offsets), see
# https://en.wikipedia.org/wiki/Sparse_matrix#Compressed_sparse_row_(CSR,_CRS_or_Yale_format)
# for more details
# This function returns a tuples of
# 1. A CSR representation of the data
# 2. A CSR represnetation of the labels
# 3. A list of list representation of the labels
def load_svm_as_csr_numpy(path, use_softmax):
    full_path = find_full_filepath(path)
    data = load_svmlight_file(full_path, multilabel=True)
    data_x = (
        data[0].indices.astype("uint32"),
        data[0].data.astype("float32"),
        data[0].indptr.astype("uint32"),
    )
    data_y = _list_of_lists_to_csr(data[1], use_softmax)
    return data_x, data_y, data[1]
