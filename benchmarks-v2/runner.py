import argparse
import os

import mlflow
from configs import bolt_configs
from configs.bolt_configs import BoltBenchmarkConfig
from dotenv import load_dotenv
from thirdai import bolt, dataset
from thirdai.experimental import MlflowCallback


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


def load_datasets(config):
    validate_dataset_attributes(config)

    result = {
        "train_data": [],
        "train_labels": [],
        "test_data": [],
        "test_labels": [],
        "test_labels_np": [],
    }

    dataset_format = config.dataset_format

    if dataset_format == "svm":
        train_dataset = dataset.load_bolt_svm_dataset(
            filename=config.train_dataset_path, batch_size=config.train_batch_size
        )
        test_dataset = dataset.load_bolt_svm_dataset(
            filename=config.test_dataset_path, batch_size=config.test_batch_size
        )


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


def define_bolt_model(config: BoltBenchmarkConfig):
    def construct_criteo_dlrm_model(config):
        pass

    def get_loss_function(config):
        loss = config.loss_fn.lower()
        if loss == "categoricalcrossentropyloss":
            return bolt.nn.losses.CategoricalCrossEntropyLoss()
        elif loss == "binarycrossentropyloss":
            return bolt.nn.losses.BinaryCrossEntropyLoss()
        elif loss == "meansquarederror":
            return bolt.nn.losses.MeanSquaredError()
        raise ValueError(f"Invalid loss function: {loss}")

    if config.dataset_name == "criteo":
        construct_criteo_dlrm_model()

    input_node = construct_input_node(benchmark_config=config)
    hidden_node = construct_fully_connected_node(benchmark_config=config)(input_node)
    output_node = construct_fully_connected_node(
        benchmark_config=config, is_hidden_layer=False
    )(hidden_node)

    loss_function = get_loss_function(config)
    model = bolt.nn.Model(inputs=input_node, output=output_node)
    model.compile(loss=loss_function, print_when_done=False)
    model.summary(detailed=True)

    return model


def get_mlflow_uri():
    load_dotenv()
    return os.getenv("MLFLOW_URI")


def launch_bolt_benchmark(config: BoltBenchmarkConfig, run_name: str) -> None:
    if not issubclass(config, BoltBenchmarkConfig):
        raise ValueError(
            f"The input config must be a bolt config. Given a config of type {config.__class__}"
        )

    mlflow_uri = get_mlflow_uri()
    # callbacks = [
    #     MlflowCallback(
    #         mlflow_uri, config.experiment_name, run_name, config.dataset_name, {}
    #     )
    # ]

    model = define_bolt_model(config)
    dataset = load_datasets(config)


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
