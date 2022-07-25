# TODO(josh): Add back mach benchmark

import argparse
from sklearn.metrics import roc_auc_score
from multiprocessing.sharedctypes import Value
import toml
from pathlib import Path
from utils import (
    start_mlflow,
    verify_mlflow_args,
    find_full_filepath,
    log_single_epoch_training_metrics,
    log_prediction_metrics,
    config_get,
)
from thirdai import bolt, dataset
import numpy as np


def main():
    parser = build_arg_parser()
    args = parser.parse_args()

    verify_mlflow_args(parser, mlflow_args=args)

    config = toml.load(args.config_path)

    model = load_and_compile_model(config)
    datasets = load_all_datasets(config)
    start_mlflow(config, mlflow_args=args)
    run_experiment(model, datasets, config, use_mlflow=not args.disable_mlflow)


def load_and_compile_model(model_config):
    name_to_node = {}

    def get_node_by_name(node_name):
        if node_name in name_to_node:
            return name_to_node[node_name]
        raise ValueError(f"{node_name} not found in previously defined nodes")

    nodes_with_no_successor = set()
    token_inputs = []
    inputs = []
    for node_config in config_get(model_config, "nodes"):
        node = construct_node(node_config)
        node_name = config_get(node_config, "name")
        node_type = config_get(node_config, "type")

        if node_type == "Input":
            inputs.append(node)
        elif node_type == "TokenInput":
            token_inputs.append(node)
        elif "pred" in node_config:
            pred_name = node_config["pred"]
            pred_node = get_node_by_name(pred_name)
            nodes_with_no_successor.remove(pred_name)
            node(pred_node)
        elif "preds" in node_config:
            pred_names = node_config["preds"]
            pred_nodes = [get_node_by_name(pred_name) for pred_name in pred_names]
            for pred_name in pred_names:
                nodes_with_no_successor.remove(pred_name)
            node(pred_nodes)
        else:
            raise ValueError(
                "Node should either be an Input/TokenInput or specify pred/preds"
            )

        nodes_with_no_successor.add(node_name)
        name_to_node[node_name] = node

    if len(nodes_with_no_successor) != 1:
        raise ValueError(
            "There should only be one output node (nodes with no successors), "
            + f"but found {len(nodes_with_no_successor)}"
        )

    output_node = name_to_node[list(nodes_with_no_successor)[0]]
    model = bolt.graph.Model(
        inputs=inputs, token_inputs=token_inputs, output=output_node
    )
    model.compile(loss=get_loss(model_config))
    return model


# Returns a map from
# ["train_data", "train_tokens", "train_labels", "test_data", "test_tokens", "test_labels"]
# to lists of datasets (except for the labels, which will either be a single dataset or None)
def load_all_datasets(dataset_config):
    result = {
        "train_data": [],
        "train_tokens": [],
        "train_labels": [],
        "test_data": [],
        "test_tokens": [],
        "test_labels": [],
    }

    all_dataset_configs = config_get(dataset_config, "datasets")
    for single_dataset_config in all_dataset_configs:
        format = config_get(single_dataset_config, "format")
        dataset_types = config_get(single_dataset_config, "type_list")

        if format == "svm":
            loaded_datasets = load_svm_dataset(single_dataset_config)
        elif format == "click":
            loaded_datasets = load_click_through_dataset(single_dataset_config)
        else:
            raise ValueError(f"{format} is an unrecognized dataformat")

        if len(dataset_types) != len(loaded_datasets):
            raise ValueError(
                f"The number of datasets loaded {len(loaded_datasets)} did "
                f"not match the number of dataset types {len(dataset_types)}"
                f"for the following config: \n {single_dataset_config}"
            )

        for dataset_type, dataset in zip(dataset_types, loaded_datasets):
            if dataset_type not in result.keys():
                raise ValueError(
                    f"The dataset type {dataset_type} is not in {result.keys()}"
                )
            result[dataset_type].append(dataset)

    for label_name in ["train_labels", "test_labels"]:
        if len(result[label_name]) > 1:
            raise ValueError(
                f"Must pass in 0 or 1 label datasets, but found {len(result[label_name])} {label_name}s"
            )
        if len(result[label_name]) == 1:
            result[label_name] = result[label_name][0]
        else:
            result[label_name] = None

    return result


def run_experiment(model, datasets, experiment_config, use_mlflow):
    num_epochs, train_config = load_train_config(experiment_config)
    predict_config = load_predict_config(experiment_config)

    for epoch_num in range(num_epochs):

        freeze_hash_table_if_needed(model, experiment_config, epoch_num)
        switch_to_sparse_inference_if_needed(
            predict_config, experiment_config, epoch_num
        )

        train_metrics = model.train(
            train_data=datasets["train_data"],
            train_tokens=datasets["train_tokens"],
            train_labels=datasets["train_labels"],
            train_config=train_config,
        )
        if use_mlflow:
            log_single_epoch_training_metrics(train_metrics)

        predict_metrics = model.predict(
            test_data=datasets["test_data"],
            test_tokens=datasets["test_tokens"],
            test_labels=datasets["test_labels"],
            predict_config=predict_config,
        )
        if use_mlflow:
            log_prediction_metrics(predict_metrics)

    if "save" in experiment_config.keys():
        model.save(config_get(experiment_config, "save"))


def construct_fully_connected_node(fc_config):
    use_default_sampling = fc_config.get("use_default_sampling", False)
    sparsity = fc_config.get("sparsity", 1)

    if use_default_sampling or sparsity == 1:
        return bolt.graph.FullyConnected(
            dim=config_get(fc_config, "dim"),
            sparsity=sparsity,
            activation=config_get(fc_config, "activation"),
        )

    return bolt.graph.FullyConnected(
        dim=config_get(fc_config, "dim"),
        sparsity=sparsity,
        activation_function=config_get(fc_config, "activation"),
        sampling_config=bolt.SamplingConfig(
            hashes_per_table=config_get(fc_config, "hashes_per_table"),
            num_tables=config_get(fc_config, "num_tables"),
            range_pow=config_get(fc_config, "range_pow"),
            reservoir_size=config_get(fc_config, "reservoir_size"),
            hash_function=fc_config.get("hash_function", "DWTA"),
        ),
    )
    
def construct_embedding_node(embedding_config):
    num_embedding_lookups = config_get(embedding_config, "num_embedding_lookups")
    lookup_size = config_get(embedding_config, "lookup_size")
    log_embedding_block_size = config_get(embedding_config, "log_embedding_block_size")
    
    return bolt.graph.Embedding(
        num_embedding_lookups=num_embedding_lookups,
        lookup_size=lookup_size,
        log_embedding_block_size=log_embedding_block_size
    )

def construct_node(node_config):
    node_type = config_get(node_config, "type")
    if node_type == "Input":
        return bolt.graph.Input(dim=config_get(node_config, "dim"))
    if node_type == "Concatenate":
        return bolt.graph.Concatenate()
    if node_type == "FullyConnected":
        return construct_fully_connected_node(node_config)
    if node_type == "TokenInput":
        return bolt.graph.TokenInput()
    if node_type == "Embedding":
        return construct_embedding_node(node_config)
    raise ValueError(f"{node_type} is not a valid node type.")


def get_loss(model_config):
    loss_string = config_get(model_config, "loss_fn")
    # TODO(josh/nick): Add an option to pass in the loss function as string to compile
    # TODO(josh): Consider moving to python 3.10 so we have the match pattern
    if loss_string.lower() == "categoricalcrossentropyloss":
        return bolt.CategoricalCrossEntropyLoss()
    if loss_string.lower() == "binarycrossentropyloss":
        return bolt.BinaryCrossEntropyLoss()
    if loss_string.lower() == "meansquarederror":
        return bolt.MeanSquaredError()
    raise ValueError(f"{loss_string} is not a valid loss function.")


def load_svm_dataset(dataset_config):
    dataset_path = find_full_filepath(config_get(dataset_config, "path"))
    return dataset.load_bolt_svm_dataset(
        dataset_path, batch_size=config_get(dataset_config, "batch_size")
    )


def load_click_through_dataset(dataset_config):
    dataset_path = find_full_filepath(config_get(dataset_config, "path"))
    return dataset.load_click_through_dataset(
        filename=dataset_path,
        batch_size=config_get(dataset_config, "batch_size"),
        num_numerical_features=config_get(dataset_config, "num_numerical_features"),
        max_categorical_features=config_get(dataset_config, "max_categorical_features"),
        delimiter=config_get(dataset_config, "delimiter")
    )

# Because of how our experiment works, we always set num_epochs=1 and return
# num_epochs as the first element of a 2 item tuple (the second element is
# the train_config)
def load_train_config(experiment_config):
    train_config = bolt.graph.TrainConfig.make(
        epochs=1, learning_rate=config_get(experiment_config, "learning_rate")
    ).with_metrics(config_get(experiment_config, "train_metrics"))
    if "reconstruct_hash_functions" in experiment_config.keys():
        train_config.with_reconstruct_hash_functions(
            experiment_config["reconstruct_hash_functions"]
        )
    if "rebuild_hash_tables" in experiment_config.keys():
        train_config.with_rebuild_hash_tables(experiment_config["rebuild_hash_tables"])
    return config_get(experiment_config, "epochs"), train_config


def load_predict_config(experiment_config):
    return bolt.graph.PredictConfig.make().with_metrics(
        config_get(experiment_config, "test_metrics")
    )

def freeze_hash_table_if_needed(model, experiment_config, current_epoch):
    should_freeze_hash_tables = (
        "freeze_hash_tables_epoch" in experiment_config.keys()
        and current_epoch == experiment_config["freeze_hash_tables_epoch"]
    )
    if should_freeze_hash_tables:
        print(f"Freezing hash tables at beginning of epoch {current_epoch}")
        model.freeze_hash_tables()


def switch_to_sparse_inference_if_needed(
    predict_config, experiment_config, current_epoch
):
    use_sparse_inference = (
        "sparse_inference_epoch" in experiment_config.keys()
        and current_epoch >= experiment_config["sparse_inference_epoch"]
    )
    if use_sparse_inference:
        print(f"Switching to sparse inference on epoch {current_epoch}")
        predict_config.enable_sparse_inference()
    return use_sparse_inference


def build_arg_parser():
    parser = argparse.ArgumentParser(
        description="Creates, trains, and tests a bolt network on the specified config."
    )

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
        "--upload_artifacts",
        action="store_true",
        help="Whether to upload artifacts to mlflow.",
    )
    return parser


if __name__ == "__main__":
    main()
