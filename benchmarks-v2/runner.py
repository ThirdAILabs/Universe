import argparse
import os

import mlflow
import numpy as np
from configs import bolt_configs
from configs.bolt_configs import BoltBenchmarkConfig
from dotenv import load_dotenv
from thirdai import bolt, dataset
from thirdai.experimental import MlflowCallback


def get_mlflow_uri():
    load_dotenv()
    return os.getenv("MLFLOW_URI")


def validate_dataset_attributes(config):
    attributes = [
        "dataset_format",
        "train_dataset_path",
        "test_dataset_path",
        "train_batch_size",
        "test_batch_size",
    ]
    for attribute in attributes:
        assert hasattr(config, attribute)


def load_wayfair_dataset(filename, batch_size, output_dim, shuffle=True):
    batch_processor = dataset.GenericBatchProcessor(
        input_blocks=[dataset.blocks.TextPairGram(col=1)],
        label_blocks=[
            dataset.blocks.NumericalId(col=0, n_classes=output_dim, delimiter=",")
        ],
        has_header=False,
        delimiter="\t",
    )

    dataloader = dataset.DatasetLoader(
        data_source=dataset.FileDataSource(filename=filename, batch_size=batch_size),
        batch_processor=batch_processor,
        shuffle=shuffle,
    )
    data, labels = dataloader.load_in_memory()
    return data, labels


def load_click_through_dataset(
    filename,
    batch_size,
    max_num_numerical_features,
    max_categorical_features,
    delimiter,
):
    data, _, labels = dataset.load_click_through_dataset(
        filename=filename,
        batch_size=batch_size,
        max_num_numerical_features=max_num_numerical_features,
        max_categorical_features=max_categorical_features,
        delimiter=delimiter,
    )
    return data, labels


def load_datasets(config):
    validate_dataset_attributes(config)

    dataset_format = config.dataset_format

    if dataset_format == "svm":
        train_data, train_labels = dataset.load_bolt_svm_dataset(
            filename=config.train_dataset_path, batch_size=config.train_batch_size
        )
        test_data, test_labels = dataset.load_bolt_svm_dataset(
            filename=config.test_dataset_path, batch_size=config.test_batch_size
        )
    elif dataset_format == "wayfair":
        assert hasattr(config, "output_dim")
        train_data, train_labels = load_wayfair_dataset(
            filename=config.train_dataset_path,
            batch_size=config.train_batch_size,
            output_dim=config.output_dim,
        )
        test_data, test_labels = load_wayfair_dataset(
            filename=config.test_dataset_path,
            batch_size=config.test_batch_size,
            output_dim=config.output_dim,
            shuffle=False,
        )
    elif dataset_format == "click_through":
        assert hasattr(config, "max_num_numerical_features")
        assert hasattr(config, "max_num_categorical_features")

        train_data, _, train_labels = load_click_through_dataset(
            filename=config.train_dataset_path,
            batch_size=config.train_batch_size,
            max_num_numerical_features=config.max_num_numerical_features,
            max_categorical_features=config.max_num_categorical_features,
            delimiter=config.delimiter,
        )

        test_data, _, test_labels = load_click_through_dataset(
            filename=config.test_dataset_path,
            batch_size=config.test_batch_size,
            max_num_numerical_features=config.max_num_numerical_features,
            max_categorical_features=config.max_categorical_features,
            delimiter=config.delimiter,
        )

    return train_data, train_labels, test_data, test_labels


def construct_input_node(benchmark_config):
    assert hasattr(benchmark_config, "input_dim")

    dimension = benchmark_config.input_dim
    return bolt.nn.Input(dim=dimension)


def construct_fully_connected_node(benchmark_config, is_hidden_layer=True):
    if is_hidden_layer:
        assert hasattr(benchmark_config, "hidden_dim")
        sparsity = benchmark_config.hidden_sparsity
        dim = benchmark_config.hidden_dim
        activation = benchmark_config.hidden_activation
        sampling_config = benchmark_config.hidden_sampling_config

    else:
        assert hasattr(benchmark_config, "output_dim")
        dim = benchmark_config.output_dim
        sparsity = benchmark_config.output_sparsity
        activation = benchmark_config.output_activation
        sampling_config = benchmark_config.output_sampling_config

    if sampling_config is None or sparsity == 1.0:
        return bolt.nn.FullyConnected(dim=dim, sparsity=sparsity, activation=activation)
    return bolt.nn.FullyConnected(
        dim=dim,
        sparsity=sparsity,
        activation=activation,
        sampling_config=sampling_config,
    )

def get_train_and_eval_configs(benchmark_config, callbacks=None):
    learning_rate = benchmark_config.learning_rate
    metrics = [benchmark_config.metric_type]

    train_config = bolt.TrainConfig(epochs=1, learning_rate=learning_rate).with_metrics(
        metrics
    )
    if callbacks is not None:
        train_config.with_callbacks(callbacks)

    if hasattr(benchmark_config, "rehashing_factor"):
        train_config.with_reconstruct_hash_functions(benchmark_config.rehashing_factor)

    if hasattr(benchmark_config, "rebuild_hash_tables_factor"):
        train_config.with_rebuild_hash_tables(
            benchmark_config.rebuild_hash_tables_factor
        )

    eval_config = bolt.EvalConfig().with_metrics(metrics)
    if benchmark_config.compute_roc_auc == True:
        eval_config.return_activations()

    return train_config, eval_config


def define_dlrm_model(config, loss_function: bolt.nn.losses):
    def get_input_nodes(nodes):
        assert "numerical_input" in nodes.keys()
        assert "categorical_input" in nodes.keys()
        assert (
            "dim" in nodes["numerical_input"].keys()
            and "dim" in nodes["categorical_input"].keys()
        )
        assert (
            "dim" in nodes["categorical_input"].keys()
            and "dim" in nodes["categorical_input"].keys()
        )
        assert (
            "min_num_tokens" in nodes["categorical_input"].keys()
            and "max_num_tokens" in nodes["categorical_input"].keys()
        )

        num_tokens_range = (
            nodes["categorical_input"]["min_num_tokens"],
            nodes["categorical_input"]["max_num_tokens"],
        )
        input_node = bolt.nn.Input(dim=nodes["numerical_input"]["dim"])
        token_input_node = bolt.nn.TokenInput(
            dim=nodes["categorical_input"]["dim"], num_tokens_range=num_tokens_range
        )
        return input_node, token_input_node

    def get_fully_connected_node(node):
        sparsity = node["sparsity"] if "sparsity" in node else 1.0
        return bolt.nn.FullyConnected(
            dim=node["dim"],
            sparsity=sparsity,
            activation=node["activation"],
        )

    def get_embedding_node(node):
        return bolt.nn.Embedding(
            num_embedding_lookups=node["num_embedding_lookups"],
            lookup_size=node["lookup_size"],
            log_embedding_block_size=node["log_embedding_block_size"],
            reduction=node["reduction"],
            num_tokens_per_input=node["num_tokens_per_input"],
        )

    assert hasattr(config, "nodes")
    nodes = config.nodes
    input_node, token_input = get_input_nodes(nodes=nodes)
    first_hidden = get_fully_connected_node(nodes["hidden1"])(input_node)
    second_hidden = get_fully_connected_node(nodes["hidden2"])(first_hidden)

    embedding = get_embedding_node(nodes["embedding"])(token_input)
    concat = bolt.nn.Concatenate()([second_hidden, embedding])
    third_hidden = get_fully_connected_node(nodes["hidden3"])(concat)

    output = get_fully_connected_node(nodes["output"])(third_hidden)

    model = bolt.nn.Model(inputs=[input_node, token_input], output=output)
    model.compile(loss=loss_function, print_when_done=False)
    model.summary(detailed=True)


def define_bolt_model(config: BoltBenchmarkConfig):
    def get_loss_function(config):
        loss = config.loss_fn.lower()
        if loss == "categoricalcrossentropyloss":
            return bolt.nn.losses.CategoricalCrossEntropy()
        elif loss == "binarycrossentropyloss":
            return bolt.nn.losses.BinaryCrossEntropy()
        elif loss == "meansquarederror":
            return bolt.nn.losses.MeanSquaredError()
        raise ValueError(f"Invalid loss function: {loss}")

    if config.dataset_name == "criteo_46m":
        loss = get_loss_function(config)
        return define_dlrm_model(config=config, loss_function=loss)

    input_node = construct_input_node(benchmark_config=config)
    hidden_node = construct_fully_connected_node(benchmark_config=config)(input_node)
    output_node = construct_fully_connected_node(
        benchmark_config=config, is_hidden_layer=False
    )(hidden_node)

    loss_function = get_loss_function(config)
    model = bolt.nn.Model(inputs=[input_node], output=output_node)
    model.compile(loss=loss_function, print_when_done=False)
    model.summary(detailed=True)

    return model


def compute_roc_auc(predict_output, test_labels_path, mlflow_callback=None):
    with open(config.train_dataset_path) as file:
        test_labels = [np.array([int(line[0]) for line in file.readlines()])]

    if len(predict_output) != 2:
        raise ValueError("Cannot compute the AUC without dense activations")

    activations = predict_output[1]
    if len(activations) != len(test_labels):
        raise ValueError(f"Length of activations must match the length of test labels")
    # If there are two output neurons then the true scores are activations of the second neuron.
    if len(activations.shape) == 2 and activations.shape[1] == 2:
        scores = activations[:, 1]
    # If there is a single output neuron the it is the true score.
    elif len(activations.shape) == 2 and activations.shape[1] == 1:
        scores = activations[:, 0]
    else:
        raise ValueError(
            "Activations must have shape (n,1) or (n,2) to compute the AUC"
        )

    from sklearn.metric import roc_auc_score

    auc = roc_auc_score(test_labels, scores)
    print(f"AUC : {auc}")

    if mlflow_callback is not None:
        mlflow_callback.log_additional_metric(key="roc_auc", value=auc)


def launch_bolt_benchmark(config: BoltBenchmarkConfig, run_name: str) -> None:
    if not issubclass(config, BoltBenchmarkConfig):
        raise ValueError(
            f"The input config must be a bolt config. Given a config of type {config.__class__}"
        )

    mlflow_uri = get_mlflow_uri()
    experiment_name = config.experiment_name
    dataset_name = config.dataset_name

    mlflow_callback = MlflowCallback(
        tracking_uri=mlflow_uri,
        experiment_name=experiment_name,
        run_name=run_name,
        dataset_name=dataset_name,
        experiment_args={},
    )
    callbacks = [mlflow_callback]
    callbacks.extend(config.callbacks)

    model = define_bolt_model(config)
    train_set, train_labels, test_set, test_labels = load_datasets(config)

    train_config, eval_config = get_train_and_eval_configs(
        benchmark_config=config, callbacks=callbacks
    )

    for _ in range(config.num_epochs):
        train_metrics = model.train(
            train_data=train_set, train_labels=train_labels, train_config=train_config
        )
        print(f"train_metrics = {train_metrics}")

        for k, v in train_metrics.items():
            mlflow_callback.log_additional_metric(key=k, value=v[0])

        predict_output = model.evaluate(
            test_data=test_set, test_labels=test_labels, eval_config=eval_config
        )
        mlflow_callback.log_additional_metric(
            key=predict_output[0][0], value=predict_output[0][1]
        )

        if config.compute_roc_auc:
            compute_roc_auc(
                predict_output=predict_output,
                test_labels_path=config.test_dataset_path,
                mlflow_callback=mlflow_callback,
            )

    if hasattr(config, "save"):
        model.save(config.save)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark a dataset with Bolt")
    parser.add_argument(
        "--run_name", default="", required=True, help="The job name to track in MLflow"
    )
    parser.add_argument(
        "--config_name",
        default="",
        required=True,
        help="The name of the config for Bolt",
    )

    arguments = parser.parse_args()
    config = getattr(bolt_configs, arguments.config_name)
    launch_bolt_benchmark(config=config, run_name=arguments.run_name)
