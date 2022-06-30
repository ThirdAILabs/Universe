import toml
import sys
import os
from thirdai import bolt, dataset
import numpy as np
from sklearn.metrics import roc_auc_score
from typing import Tuple, Any, Optional, Dict, List
import socket
import platform
import psutil
import mlflow
import argparse
from utils import log_config_info, log_machine_info, start_mlflow


def log_training_metrics(metrics: Dict[str, List[float]]):
    # In this benchmarking script, train is only called with one epoch
    # at a time so we can simplify the logging for mlflow in this way.
    mlflow_metrics = {k: v[0] for k, v in metrics.items()}
    mlflow.log_metrics(mlflow_metrics)


def create_fully_connected_layer_configs(
    configs: List[Dict[str, Any]]
) -> List[bolt.FullyConnected]:
    layers = []
    for config in configs:

        if config.get("use_default_sampling", False):
            layer = bolt.FullyConnected(
                dim=config.get("dim"),
                sparsity=config.get("sparsity", 1.0),
                activation_function=bolt.getActivationFunction(
                    config.get("activation")
                ),
            )
        else:
            layer = bolt.FullyConnected(
                dim=config.get("dim"),
                sparsity=config.get("sparsity", 1.0),
                activation_function=bolt.getActivationFunction(
                    config.get("activation")
                ),
                sampling_config=bolt.SamplingConfig(
                    hashes_per_table=config.get("hashes_per_table", 0),
                    num_tables=config.get("num_tables", 0),
                    range_pow=config.get("range_pow", 0),
                    reservoir_size=config.get("reservoir_size", 128),
                    hash_function=config.get("hash_function", "DWTA"),
                ),
            )

        layers.append(layer)
    return layers


def create_embedding_layer_config(config: Dict[str, Any]) -> bolt.Embedding:
    return bolt.Embedding(
        num_embedding_lookups=config.get("num_embedding_lookups"),
        lookup_size=config.get("lookup_size"),
        log_embedding_block_size=config.get("log_embedding_block_size"),
    )


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


def load_dataset(
    config: Dict[str, Any]
) -> Optional[
    Tuple[
        dataset.BoltDataset,  # train_x
        dataset.BoltDataset,  # train_y
        dataset.BoltDataset,  # test_x
        dataset.BoltDataset,  # test_y
    ]
]:
    train_filename = find_full_filepath(config["dataset"]["train_data"])
    test_filename = find_full_filepath(config["dataset"]["test_data"])
    batch_size = config["params"]["batch_size"]
    if config["dataset"]["format"].lower() == "svm":
        train_x, train_y = dataset.load_bolt_svm_dataset(train_filename, batch_size)
        test_x, test_y = dataset.load_bolt_svm_dataset(test_filename, batch_size)
        return train_x, train_y, test_x, test_y
    elif config["dataset"]["format"].lower() == "csv":
        delimiter = config["dataset"].get("delimeter", ",")
        train_x, train_y = dataset.load_bolt_csv_dataset(
            train_filename, batch_size, delimiter
        )
        test_x, test_y = dataset.load_bolt_csv_dataset(
            test_filename, batch_size, delimiter
        )
        return train_x, train_y, test_x, test_y
    else:
        print("Invalid dataset format specified")
        return None


def load_click_through_dataset(
    config: Dict[str, Any], sparse_labels: bool
) -> Tuple[
    dataset.ClickThroughDataset,  # train_x
    dataset.BoltDataset,  # train_y
    dataset.ClickThroughDataset,  # test_x
    dataset.BoltDataset,  # test_y
]:
    train_filename = find_full_filepath(config["dataset"]["train_data"])
    test_filename = find_full_filepath(config["dataset"]["test_data"])
    batch_size = config["params"]["batch_size"]
    dense_features = config["dataset"]["dense_features"]
    categorical_features = config["dataset"]["categorical_features"]
    train_x, train_y = dataset.load_click_through_dataset(
        train_filename, batch_size, dense_features, categorical_features, sparse_labels
    )
    test_x, test_y = dataset.load_click_through_dataset(
        test_filename, batch_size, dense_features, categorical_features, sparse_labels
    )
    return train_x, train_y, test_x, test_y


def get_labels(dataset: str):
    labels = []
    with open(find_full_filepath(dataset)) as file:
        for line in file.readlines():
            items = line.strip().split()
            label = int(items[0])
            labels.append(label)
    return np.array(labels)


def train_fcn(config: Dict[str, Any], mlflow_enabled: bool):
    layers = create_fully_connected_layer_configs(config["layers"])
    input_dim = config["dataset"]["input_dim"]
    network = bolt.Network(layers=layers, input_dim=input_dim)

    learning_rate = config["params"]["learning_rate"]
    epochs = config["params"]["epochs"]
    max_test_batches = config["dataset"].get("max_test_batches", None)
    rehash = config["params"]["rehash"]
    rebuild = config["params"]["rebuild"]
    use_sparse_inference = "sparse_inference_epoch" in config["params"].keys()
    if use_sparse_inference:
        sparse_inference_epoch = config["params"]["sparse_inference_epoch"]
    else:
        sparse_inference_epoch = None

    data = load_dataset(config)
    if data is None:
        raise ValueError("Unable to load a dataset. Please check the config")

    train_x, train_y, test_x, test_y = data

    if config["params"]["loss_fn"].lower() == "categoricalcrossentropyloss":
        loss = bolt.CategoricalCrossEntropyLoss()
    elif config["params"]["loss_fn"].lower() == "meansquarederror":
        loss = bolt.MeanSquaredError()
    else:
        print("'{}' is not a valid loss function".format(config["params"]["loss_fn"]))
        return

    train_metrics = config["params"]["train_metrics"]
    test_metrics = config["params"]["test_metrics"]

    for e in range(epochs):
        # Use keyword arguments to skip batch_size parameter.
        metrics = network.train(
            train_data=train_x,
            train_labels=train_y,
            loss_fn=loss,
            learning_rate=learning_rate,
            epochs=1,
            rehash=rehash,
            rebuild=rebuild,
            metrics=train_metrics,
        )
        if mlflow_enabled:
            log_training_metrics(metrics)

        if use_sparse_inference and e == sparse_inference_epoch:
            network.enable_sparse_inference()

        if max_test_batches is None:
            # Use keyword arguments to skip batch_size parameter.
            metrics, _ = network.predict(
                test_data=test_x, test_labels=test_y, metrics=test_metrics
            )
            if mlflow_enabled:
                mlflow.log_metrics(metrics)
        else:
            # Use keyword arguments to skip batch_size parameter.
            metrics, _ = network.predict(
                test_data=test_x,
                test_labels=test_y,
                metrics=test_metrics,
                verbose=True,
                batch_limit=max_test_batches,
            )
            if mlflow_enabled:
                mlflow.log_metrics(metrics)
    if not max_test_batches is None:
        # If we limited the number of test batches during training we run on the whole test set at the end.
        # Use keyword arguments to skip batch_size parameter.
        metrics, _ = network.predict(
            test_data=test_x, test_labels=test_y, metrics=test_metrics
        )
        if mlflow_enabled:
            mlflow.log_metrics(metrics)

    if "save_for_inference" in config["params"].keys():
        network.save_for_inference(config["params"]["save_for_inference"])


def train_dlrm(config: Dict[str, Any], mlflow_enabled: bool):
    embedding_layer = create_embedding_layer_config(config["embedding_layer"])
    bottom_mlp = create_fully_connected_layer_configs(config["bottom_mlp_layers"])
    top_mlp = create_fully_connected_layer_configs(config["top_mlp_layers"])
    input_dim = config["dataset"]["dense_features"]
    dlrm = bolt.DLRM(
        embedding_layer=embedding_layer,
        bottom_mlp=bottom_mlp,
        top_mlp=top_mlp,
        input_dim=input_dim,
    )

    learning_rate = config["params"]["learning_rate"]
    epochs = config["params"]["epochs"]
    rehash = config["params"]["rehash"]
    rebuild = config["params"]["rebuild"]

    if config["params"]["loss_fn"].lower() == "categoricalcrossentropyloss":
        loss = bolt.CategoricalCrossEntropyLoss()
    elif config["params"]["loss_fn"].lower() == "meansquarederror":
        loss = bolt.MeanSquaredError()

    train_metrics = config["params"]["train_metrics"]
    test_metrics = config["params"]["test_metrics"]

    use_auc = "use_auc" in config["params"].keys() and config["params"].get(
        "use_auc", False
    )

    use_sparse_labels = config["top_mlp_layers"][-1]["dim"] > 1
    train_x, train_y, test_x, test_y = load_click_through_dataset(
        config, use_sparse_labels
    )
    labels = get_labels(config["dataset"]["test_data"])

    for _ in range(epochs):
        metrics = dlrm.train(
            train_x,
            train_y,
            loss,
            learning_rate,
            1,
            rehash,
            rebuild,
            train_metrics,
        )
        if mlflow_enabled:
            log_training_metrics(metrics)

        metrics, scores = dlrm.predict(test_x, test_y, test_metrics)
        if mlflow_enabled:
            mlflow.log_metrics(metrics)

        if len(scores.shape) == 2:
            preds = np.argmax(scores, axis=1)
            acc = np.mean(preds == labels)
            print("Accuracy: ", acc)

        if use_auc and len(scores.shape) == 1:
            auc = roc_auc_score(labels, scores)
            if mlflow_enabled:
                mlflow.log_metric("auc", auc)
            print("AUC: ", auc)
        elif use_auc and len(scores.shape) == 2 and scores.shape[1] == 2:
            auc = roc_auc_score(labels, scores[:, 1])
            if mlflow_enabled:
                mlflow.log_metric("auc", auc)
            print("AUC: ", auc)


def is_dlrm(config: Dict[str, Any]) -> bool:
    return "bottom_mlp_layers" in config.keys() and "top_mlp_layers" in config.keys()


def is_fcn(config: Dict[str, Any]) -> bool:
    return "layers" in config.keys()


def build_arg_parser():
    parser = argparse.ArgumentParser(
        description="Runs creates and trains a bolt network on the specified config."
    )

    parser.add_argument(
        "config_file",
        type=str,
        help="Name of the config file to use to run experiment.",
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
        help="The name of the run to use in mlflow, if mlflow is not disabled this is required.",
    )

    return parser


def main():
    parser = build_arg_parser()
    args = parser.parse_args()

    mlflow_enabled = not args.disable_mlflow

    if mlflow_enabled and not args.run_name:
        parser.print_usage()
        raise ValueError("Error: --run_name is required when using mlflow logging.")

    config_filename = sys.argv[1]
    config = toml.load(config_filename)

    if mlflow_enabled:
        experiment_name = config["job"]
        dataset = config["dataset"]["train_data"].split("/")[-1]
        start_mlflow(experiment_name, args.run_name, dataset)
        # TODO(vihan): Get the credential authentication working in github actions
        if not args.disable_upload_artifacts:
            mlflow.log_artifact(config_filename)
        log_machine_info()
        log_config_info(config)

    if is_fcn(config):
        train_fcn(config, mlflow_enabled)
    elif is_dlrm(config):
        train_dlrm(config, mlflow_enabled)
    else:
        print("Invalid network architecture specified")

    mlflow.end_run()


if __name__ == "__main__":
    main()
