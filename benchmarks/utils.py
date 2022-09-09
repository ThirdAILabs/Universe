import argparse
import mlflow
import os
import toml
import platform
import psutil
import socket
import sys
import numpy as np
from urllib.request import urlopen

from typing import Any, Dict
from sklearn.datasets import load_svmlight_file

# See https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/instancedata-data-retrieval.html
AWS_METADATA_URL = "http://169.254.169.254/latest/meta-data/public-ipv4"


def build_arg_parser(description):
    parser = argparse.ArgumentParser(description)
    parser.add_argument(
        "config_path",
        type=str,
        help="Path to a config file containing the dataset, experiment, and model configs.",
    )
    parser.add_argument(
        "--disable_mlflow",
        action="store_true",
        help="Disable mlflow logging for the current run.",
    )
    parser.add_argument(
        "--disable_upload_artifacts",
        action="store_true",
        help="Disable the mlflow artifact file logging for the current run.",
    )
    parser.add_argument(
        "--run_name",
        default="",
        type=str,
        help="The name of the run to use in mlflow. If mlflow is enabled this is required.",
    )
    parser.add_argument(
        "--log-to-stderr",
        action="store_true",
        help="Logs to stderr, based on the log-level. Use --log-level to control granularity.",
    )
    parser.add_argument(
        "--log-file",
        type=str,
        help="File to write on disk to. Leaving empty (default) implies no logging to file.",
        default="",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        help="Log level to configure.",
        default="info",
        choices=["off", "critical", "error", "warn", "info", "debug", "trace"],
    )
    return parser


def start_experiment(description):
    parser = build_arg_parser(description)
    args = parser.parse_args()
    verify_mlflow_args(parser, mlflow_args=args)
    config = load_config(args)
    return config, args


def start_mlflow(config, mlflow_args):
    if not mlflow_args.disable_mlflow:
        experiment_name = config["experiment_identifier"]
        dataset_name = config["dataset_identifier"]
        model_name = config["model_identifier"]
        start_mlflow_helper(
            experiment_name, mlflow_args.run_name, dataset_name, model_name
        )
        log_machine_info()
        if not mlflow_args.disable_upload_artifacts:
            mlflow.log_artifact(mlflow_args.config_path)


def start_mlflow_helper(experiment_name, run_name, dataset, model_name):
    file_dir = os.path.dirname(os.path.abspath(__file__))
    file_name = os.path.join(file_dir, "config.toml")
    with open(file_name) as f:
        parsed_config = toml.load(f)
        mlflow.set_tracking_uri(parsed_config["tracking"]["uri"])

    mlflow.set_experiment(experiment_name)
    mlflow.start_run(
        run_name=run_name,
        tags={"dataset": dataset, "model": model_name},
    )


def log_params(params, mlflow_args):
    if not mlflow_args.disable_mlflow:
        mlflow.log_params(params)


def log_metrics(metrics, mlflow_args):
    if not mlflow_args.disable_mlflow:
        mlflow.log_metrics(metrics)


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


def load_config(args):
    return toml.load(args.config_path)


def mlflow_is_enabled(args):
    return not args.disable_mlflow


def config_get_required(config, field):
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
    if os.path.exists(filename):
        return filename

    # Load path prefixes to look for datasets from config file in the repository.
    data_path_file = (
        os.path.dirname(os.path.abspath(__file__)) + "/../dataset_paths.toml"
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


# See https://stackoverflow.com/questions/29573081/check-if-python-script-is-running-on-an-aws-instance
# This also works in docker images running on aws.
def is_ec2_instance():
    """Check if an instance is running on EC2 by trying to retrieve ec2 metadata."""
    result = False
    try:
        result = urlopen(AWS_METADATA_URL).status == 200
    except ConnectionError:
        return result
    return result
