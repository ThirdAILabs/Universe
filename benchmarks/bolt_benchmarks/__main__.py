# TODO(josh): Add back mach benchmark

import numpy as np
import pathlib
import sys

from thirdai import bolt, dataset
from thirdai import setup_logging

from ..utils import (
    start_experiment,
    start_mlflow,
    find_full_filepath,
    log_single_epoch_training_metrics,
    log_prediction_metrics,
    mlflow_is_enabled,
    config_get_required,
    is_ec2_instance,
)


def main():

    config, args = start_experiment(
        description="Creates, trains, and tests a bolt network on the specified config."
    )

    setup_logging(
        log_to_stderr=args.log_to_stderr, path=args.log_file, level=args.log_level
    )

    model = load_and_compile_model(config)
    datasets = load_all_datasets(config)
    start_mlflow(config, mlflow_args=args)
    run_experiment(model, datasets, config, use_mlflow=mlflow_is_enabled(args))


def load_and_compile_model(model_config):
    name_to_node = {}

    def get_node_by_name(node_name):
        if node_name in name_to_node:
            return name_to_node[node_name]
        raise ValueError(f"{node_name} not found in previously defined nodes")

    nodes_with_no_successor = set()
    inputs = []
    for node_config in config_get_required(model_config, "nodes"):
        node = construct_node(node_config)
        node_name = config_get_required(node_config, "name")
        node_type = config_get_required(node_config, "type")

        if node_type == "Input":
            inputs.append(node)
        elif "pred" in node_config:
            pred_name = node_config["pred"]
            pred_node = get_node_by_name(pred_name)
            nodes_with_no_successor.remove(pred_name)
            node(pred_node)
        elif "preds" in node_config:
            pred_names = node_config["preds"]
            pred_nodes = [get_node_by_name(pred_name) for pred_name in pred_names]
            for pred_name in pred_names:
                if pred_name in nodes_with_no_successor:
                    nodes_with_no_successor.remove(pred_name)
            if config_get_required(node_config, "type") == "Switch":
                node(pred_nodes[0], pred_nodes[1])
            elif config_get_required(node_config, "type") == "DlrmAttention":
                node(pred_nodes[0], pred_nodes[1])
            else:
                node(pred_nodes)
        else:
            raise ValueError("Node should either be an Input or specify pred/preds")

        nodes_with_no_successor.add(node_name)
        name_to_node[node_name] = node

    if len(nodes_with_no_successor) != 1:
        raise ValueError(
            "There should only be one output node (nodes with no successors), "
            + f"but found {len(nodes_with_no_successor)}"
        )

    output_node = name_to_node[list(nodes_with_no_successor)[0]]
    model = bolt.graph.Model(inputs=inputs, output=output_node)
    model.compile(loss=get_loss(model_config), print_when_done=False)
    model.summary(detailed=True)
    return model


# Returns a map from
# ["train_data", "train_labels", "test_data", "test_labels"]
# to lists of datasets (except for the labels, which will either be a single dataset or None)
def load_all_datasets(dataset_config):
    # We have separate test_labels and test_labels_np so that we can load the
    # test labels both as a bolt dataset for predict and also as a numpy array
    # which is needed to compute the roc_auc.
    result = {
        "train_data": [],
        "train_labels": [],
        "test_data": [],
        "test_labels": [],
        "test_labels_np": [],
    }

    all_dataset_configs = config_get_required(dataset_config, "datasets")
    for single_dataset_config in all_dataset_configs:
        format = config_get_required(single_dataset_config, "format")
        use_s3 = single_dataset_config.get("use_s3_on_aws", False) and is_ec2_instance()
        dataset_types = config_get_required(single_dataset_config, "type_list")

        if format == "svm":
            loaded_datasets = load_svm_dataset(single_dataset_config, use_s3)
        elif format == "click":
            loaded_datasets = load_click_through_dataset(single_dataset_config, use_s3)
        elif format == "click_labels":
            loaded_datasets = load_click_through_labels(single_dataset_config, use_s3)
        elif format == "mlm_with_tokens":
            loaded_datasets = load_mlm_datasets(
                single_dataset_config, use_s3, return_tokens=True
            )
        elif format == "mlm_without_tokens":
            loaded_datasets = load_mlm_datasets(
                single_dataset_config, return_tokens=False
            )
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

    if len(result["train_labels"]) != 1:
        raise ValueError(
            f"Must have 1 train label dataset but found {len(result['train_labels'])} train_labels."
        )
    result["train_labels"] = result["train_labels"][0]

    check_test_labels(result, "test_labels")
    check_test_labels(result, "test_labels_np")

    return result


def run_experiment(model, datasets, experiment_config, use_mlflow):
    num_epochs, train_config = load_train_config(experiment_config)
    predict_config = load_predict_config(experiment_config)
    if should_compute_roc_auc(experiment_config):
        predict_config.return_activations()

    for epoch_num in range(num_epochs):

        freeze_hash_table_if_needed(model, experiment_config, epoch_num)
        switch_to_sparse_inference_if_needed(
            predict_config, experiment_config, epoch_num
        )

        train_metrics = model.train(
            train_data=datasets["train_data"],
            train_labels=datasets["train_labels"],
            train_config=train_config,
        )
        if use_mlflow:
            log_single_epoch_training_metrics(train_metrics)

        predict_output = model.predict(
            test_data=datasets["test_data"],
            test_labels=datasets["test_labels"],
            predict_config=predict_config,
        )
        if use_mlflow:
            log_prediction_metrics(predict_output)

        if should_compute_roc_auc(experiment_config):
            compute_roc_auc(predict_output, datasets, use_mlflow)

    if "save" in experiment_config.keys():
        model.save(config_get_required(experiment_config, "save"))


def get_sampling_config(layer_config):
    if layer_config.get("use_random_sampling", False):
        return bolt.RandomSamplingConfig()
    return bolt.SamplingConfig(
        hashes_per_table=config_get_required(layer_config, "hashes_per_table"),
        num_tables=config_get_required(layer_config, "num_tables"),
        range_pow=config_get_required(layer_config, "range_pow"),
        reservoir_size=config_get_required(layer_config, "reservoir_size"),
        hash_function=layer_config.get("hash_function", "DWTA"),
    )


def construct_input_node(input_config):
    dim = config_get_required(input_config, "dim")
    if (
        "min_num_tokens" in input_config.keys()
        and "max_num_tokens" in input_config.keys()
    ):
        num_tokens_range = (
            config_get_required(input_config, "min_num_tokens"),
            config_get_required(input_config, "max_num_tokens"),
        )
        return bolt.graph.TokenInput(dim=dim, num_tokens_range=num_tokens_range)
    return bolt.graph.Input(dim=dim)


def construct_fully_connected_node(fc_config):
    use_default_sampling = fc_config.get("use_default_sampling", False)
    sparsity = fc_config.get("sparsity", 1)

    if use_default_sampling or sparsity == 1:
        layer = bolt.graph.FullyConnected(
            dim=config_get_required(fc_config, "dim"),
            sparsity=sparsity,
            activation=config_get_required(fc_config, "activation"),
        )
    else:
        layer = bolt.graph.FullyConnected(
            dim=config_get_required(fc_config, "dim"),
            sparsity=sparsity,
            activation=config_get_required(fc_config, "activation"),
            sampling_config=get_sampling_config(fc_config),
        )

    if fc_config.get("use_sparse_sparse_optimization", False):
        layer.enable_sparse_sparse_optimization()

    return layer


def construct_embedding_node(embedding_config):
    num_embedding_lookups = config_get_required(
        embedding_config, "num_embedding_lookups"
    )
    lookup_size = config_get_required(embedding_config, "lookup_size")
    log_embedding_block_size = config_get_required(
        embedding_config, "log_embedding_block_size"
    )
    reduction = config_get_required(embedding_config, "reduction")
    num_tokens_per_input = embedding_config.get("num_tokens_per_input", None)

    return bolt.graph.Embedding(
        num_embedding_lookups=num_embedding_lookups,
        lookup_size=lookup_size,
        log_embedding_block_size=log_embedding_block_size,
        reduction=reduction,
        num_tokens_per_input=num_tokens_per_input,
    )


def construct_switch_node(switch_config):
    use_default_sampling = switch_config.get("use_default_sampling", False)
    sparsity = switch_config.get("sparsity", 1)

    if use_default_sampling or sparsity == 1:
        return bolt.graph.Switch(
            dim=config_get_required(switch_config, "dim"),
            sparsity=sparsity,
            activation=config_get_required(switch_config, "activation"),
            n_layers=config_get_required(switch_config, "n_layers"),
        )

    return bolt.graph.Switch(
        dim=config_get_required(switch_config, "dim"),
        sparsity=sparsity,
        activation_function=config_get_required(switch_config, "activation"),
        sampling_config=get_sampling_config(switch_config),
        n_layers=config_get_required(switch_config, "n_layers"),
    )

def construct_dlrm_attention_node(node_config):
    return bolt.graph.DlrmAttention()

def construct_node(node_config):
    node_type = config_get_required(node_config, "type")
    if node_type == "Input":
        return construct_input_node(node_config)
    if node_type == "Concatenate":
        return bolt.graph.Concatenate()
    if node_type == "FullyConnected":
        return construct_fully_connected_node(node_config)
    if node_type == "Embedding":
        return construct_embedding_node(node_config)
    if node_type == "Switch":
        return construct_switch_node(node_config)
    if node_type == "DlrmAttention":
        return construct_dlrm_attention_node(node_config);
    raise ValueError(f"{node_type} is not a valid node type.")


def get_loss(model_config):
    loss_string = config_get_required(model_config, "loss_fn").lower()
    # TODO(josh/nick): Add an option to pass in the loss function as string to compile
    # TODO(josh): Consider moving to python 3.10 so we have the match pattern
    if loss_string == "categoricalcrossentropyloss" or loss_string == "cce":
        return bolt.CategoricalCrossEntropyLoss()
    if loss_string == "binarycrossentropyloss" or loss_string == "bce":
        return bolt.BinaryCrossEntropyLoss()
    if loss_string == "meansquarederror" or loss_string == "mse":
        return bolt.MeanSquaredError()
    raise ValueError(f"{loss_string} is not a valid loss function.")


def check_test_labels(datasets_map, key):
    if len(datasets_map[key]) == 1:
        datasets_map[key] = datasets_map[key][0]
    elif len(datasets_map[key]) == 0:
        datasets_map[key] = None
    else:
        raise ValueError(
            f"Must have 0 or 1 test label datasets but found {len(datasets_map[key])} test_labels."
        )


def load_svm_dataset(dataset_config, use_s3):
    batch_size = config_get_required(dataset_config, "batch_size")
    if use_s3:
        print("Using S3 to load SVM dataset", flush=True)
        s3_prefix = "share/data/" + dataset_config["path"]
        s3_bucket = "thirdai-corp"
        data_loader = dataset.S3DataLoader(
            bucket_name=s3_bucket, prefix_filter=s3_prefix, batch_size=batch_size
        )
        return dataset.load_bolt_svm_dataset(data_loader)
    else:
        dataset_path = find_full_filepath(config_get_required(dataset_config, "path"))
        return dataset.load_bolt_svm_dataset(dataset_path, batch_size=batch_size)


def load_click_through_dataset(dataset_config, use_s3):
    if use_s3:
        raise ValueError("S3 not supported yet for loading click through datasets")

    dataset_path = find_full_filepath(config_get_required(dataset_config, "path"))
    return dataset.load_click_through_dataset(
        filename=dataset_path,
        batch_size=config_get_required(dataset_config, "batch_size"),
        max_num_numerical_features=config_get_required(
            dataset_config, "max_num_numerical_features"
        ),
        max_categorical_features=config_get_required(
            dataset_config, "max_categorical_features"
        ),
        delimiter=config_get_required(dataset_config, "delimiter"),
    )


def load_click_through_labels(dataset_config, use_s3):
    if use_s3:
        raise ValueError("S3 not supported yet for loading click through labels")

    dataset_path = find_full_filepath(config_get_required(dataset_config, "path"))
    with open(dataset_path) as file:
        return [np.array([int(line[0]) for line in file.readlines()])]


def load_mlm_datasets(dataset_config, use_s3, return_tokens):
    if use_s3:
        raise ValueError("S3 not supported yet for loading mlm datasets")

    # We load the train and test data at the same time because the need to use
    # the same loader to ensure that the words in the vocabulary are mapped to
    # the same output neuron.
    train_path = find_full_filepath(config_get_required(dataset_config, "train_path"))
    test_path = find_full_filepath(config_get_required(dataset_config, "test_path"))

    mlm_loader = dataset.MLMDatasetLoader(
        pairgram_range=config_get_required(dataset_config, "pairgram_range")
    )

    batch_size = config_get_required(dataset_config, "batch_size")

    train_data = mlm_loader.load(filename=train_path, batch_size=batch_size)

    test_data = mlm_loader.load(filename=test_path, batch_size=batch_size)

    if return_tokens:
        return train_data + test_data

    return train_data[0], train_data[2], test_data[0], test_data[2]


# Because of how our experiment works, we always set num_epochs=1 and return
# num_epochs as the first element of a 2 item tuple (the second element is
# the train_config)
def load_train_config(experiment_config):
    train_config = bolt.graph.TrainConfig.make(
        epochs=1, learning_rate=config_get_required(experiment_config, "learning_rate")
    ).with_metrics(config_get_required(experiment_config, "train_metrics"))
    if "reconstruct_hash_functions" in experiment_config.keys():
        train_config.with_reconstruct_hash_functions(
            experiment_config["reconstruct_hash_functions"]
        )
    if "rebuild_hash_tables" in experiment_config.keys():
        train_config.with_rebuild_hash_tables(experiment_config["rebuild_hash_tables"])
    return config_get_required(experiment_config, "epochs"), train_config


def load_predict_config(experiment_config):
    return bolt.graph.PredictConfig.make().with_metrics(
        config_get_required(experiment_config, "test_metrics")
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


def should_compute_roc_auc(experiment_config):
    return experiment_config.get("compute_roc_auc", False)


def compute_roc_auc(predict_output, datasets, use_mlflow):
    if datasets["test_labels_np"] is None:
        raise ValueError("Cannot compute roc_auc without test_labels_np specified.")

    if len(predict_output) != 2:
        raise ValueError("Cannot compute roc_auc without dense activations.")

    labels = datasets["test_labels_np"]
    activations = predict_output[1]

    if len(activations) != len(labels):
        raise ValueError(
            f"Length of activations must match length of test_labels_np to compute roc_auc."
        )

    # If there are two output neurons then the true scores are activations of the second neuron.
    if len(activations.shape) == 2 and activations.shape[1] == 2:
        scores = activations[:, 1]
    # If there is a single output neuron the it is the true score.
    elif len(activations.shape) == 2 and activations.shape[1] == 1:
        scores = activations[:, 0]
    else:
        raise ValueError(
            "Activations must have shape (n,1), or (n,2) to compute roc_auc."
        )

    from sklearn.metrics import roc_auc_score

    roc_auc = roc_auc_score(labels, scores)
    print(f"ROC AUC = {roc_auc}")
    if use_mlflow:
        log_prediction_metrics([{"roc_auc": roc_auc}])


if __name__ == "__main__":
    main()
