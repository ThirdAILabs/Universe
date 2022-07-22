# TODO(josh): Add back mach benchmark

import argparse
import toml
from pathlib import Path
from utils import (
    start_mlflow,
    verify_mlflow_args,
    find_full_filepath,
    log_single_epoch_training_metrics,
    log_prediction_metrics,
)
from thirdai import bolt, dataset


def construct_fully_connected_node(fc_config):
    use_default_sampling = fc_config.get("use_default_sampling", False)
    sparsity = fc_config.get("sparsity", 1)

    if use_default_sampling or sparsity == 1:
        return bolt.graph.FullyConnected(
            dim=fc_config["dim"],
            sparsity=sparsity,
            activation=fc_config["activation"],
        )

    return bolt.graph.FullyConnected(
        dim=fc_config["dim"],
        sparsity=sparsity,
        activation_function=fc_config["activation"],
        sampling_config=bolt.SamplingConfig(
            hashes_per_table=fc_config["hashes_per_table"],
            num_tables=fc_config["num_tables"],
            range_pow=fc_config["range_pow"],
            reservoir_size=fc_config["reservoir_size"],
            hash_function=fc_config.get("hash_function", "DWTA"),
        ),
    )


def construct_node(node_config):
    node_type = node_config["type"]
    if node_type == "Input":
        return bolt.graph.Input(dim=node_config["dim"])
    if node_type == "TokenInput":
        return bolt.graph.TokenInput()
    if node_type == "Concatenate":
        return bolt.graph.Concatenate()
    if node_type == "FullyConnected":
        return construct_fully_connected_node(node_config)
    raise ValueError(f"{node_type} is not a valid node type.")


def get_loss(model_config):
    loss_string = model_config["loss_fn"]
    # TODO(josh/nick): Add an option to pass in the loss function as string to compile
    # TODO(josh): Consider moving to python 3.10 so we have the match pattern
    if loss_string.lower() == "categoricalcrossentropyloss":
        return bolt.CategoricalCrossEntropyLoss()
    if loss_string.lower() == "binarycrossentropyloss":
        return bolt.BinaryCrossEntropyLoss()
    if loss_string.lower() == "meansquarederror":
        return bolt.MeanSquaredError()
    raise ValueError(f"{loss_string} is not a valid loss function.")


def load_and_compile_model(model_config):
    name_to_node = {}
    nodes_with_no_successor = set()
    token_inputs = []
    inputs = []
    for node_config in model_config["nodes"]:
        node = construct_node(node_config)
        node_name = node_config["name"]
        node_type = node_config["type"]

        if node_type == "Input":
            inputs.append(node)
        elif node_type == "TokenInput":
            token_inputs.append(node)
        elif "pred" in node_config:
            pred_name = node_config["pred"]
            pred_node = name_to_node[pred_name]
            nodes_with_no_successor.remove(pred_name)
            node(pred_node)
        elif "preds" in node_config:
            pred_names = node_config["preds"]
            pred_nodes = [name_to_node[pred_name] for pred_name in pred_names]
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


def load_svm_dataset(dataset_config):
    dataset_path = find_full_filepath(dataset_config["path"])
    return dataset.load_bolt_svm_dataset(dataset_path, dataset_config["batch_size"])


def load_clickthrough_dataset(dataset_config):
    batch_size = dataset_config["batch_size"]
    dataset_path = find_full_filepath(dataset_config["path"])
    num_numerical_features = dataset_config["num_numerical_features"]
    num_categorical_features = dataset_config["num_categorical_features"]
    categorical_labels = dataset_config["categorical_labels"]
    return dataset.load_click_through_dataset(
        dataset_path,
        batch_size,
        num_numerical_features,
        num_categorical_features,
        categorical_labels,
    )


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

    for single_dataset_config in dataset_config["datasets"]:
        format = single_dataset_config["format"]
        dataset_types = single_dataset_config["type_list"]

        if format == "svm":
            loaded_datasets = load_svm_dataset(single_dataset_config)
        elif format == "click":
            loaded_datasets = load_clickthrough_dataset(single_dataset_config)
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


# Because of how our experiment works, we always set num_epochs=1 and return
# num_epochs as the first element of a 2 item tuple (the second element is
# the train_config)
def load_train_config(experiment_config):
    train_config = bolt.graph.TrainConfig.make(
        epochs=1, learning_rate=experiment_config["learning_rate"]
    ).with_metrics(experiment_config["train_metrics"])
    if "rehash" in experiment_config.keys():
        train_config.with_reconstruct_hash_functions(experiment_config["rehash"])
    if "rebuild" in experiment_config.keys():
        train_config.with_rebuild_hash_tables(experiment_config["rebuild"])
    return experiment_config["epochs"], train_config


def load_predict_config(experiment_config):
    predict_config = bolt.graph.PredictConfig.make().with_metrics(
        experiment_config["test_metrics"]
    )
    if "max_test_batches" in experiment_config.keys():
        predict_config.with_max_test_batches(experiment_config["max_test_batches"])
    return predict_config


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

    # If we limited the number of test batches during training we want to run on the whole test set at the end.
    if "max_test_batches" in experiment_config.keys():
        predict_config.with_max_test_batches(2**64)
        predict_metrics = model.predict(
            test_data=datasets["train_data"],
            test_tokens=datasets["train_token_data"],
            test_labels=datasets["train_labels"],
            predict_config=predict_config,
        )
        if use_mlflow:
            log_prediction_metrics(predict_metrics)

    if "save" in experiment_config.keys():
        model.save(experiment_config["save"])


def build_arg_parser():
    parser = argparse.ArgumentParser(
        description="Creates, trains, and tests a bolt network on the specified config."
    )

    parser.add_argument(
        "config_folder",
        type=str,
        help="Path to a config folder containing the dataset, experiment, and model configs.",
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
        "--dataset_config_path",
        default="dataset.txt",
        type=str,
        help="Relative path to the dataset config in the config_folder"
    )
    parser.add_argument(
        "--experiment_config_path",
        default="experiment.txt",
        type=str,
        help="Relative path to the experiment config in the config_folder"
    )
    parser.add_argument(
        "--model_config_path",
        default="model.txt",
        type=str,
        help="Relative path to the model config in the config_folder"
    )


    return parser


def main():
    parser = build_arg_parser()
    args = parser.parse_args()

    verify_mlflow_args(parser, mlflow_args=args)

    config_folder = Path(args.config_folder)
    model_config_filename = config_folder / args.model_config_path
    dataset_config_filename = config_folder / args.dataset_config_path
    experiment_config_filename = config_folder / args.experiment_config_path
    model_config = toml.load(model_config_filename)
    dataset_config = toml.load(dataset_config_filename)
    experiment_config = toml.load(experiment_config_filename)

    model = load_and_compile_model(model_config)
    datasets = load_all_datasets(dataset_config)
    start_mlflow(model_config, dataset_config, experiment_config, mlflow_args=args)
    run_experiment(
        model, datasets, experiment_config, use_mlflow=not args.disable_mlflow
    )


if __name__ == "__main__":
    main()
