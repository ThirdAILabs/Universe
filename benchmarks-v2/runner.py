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

def load_wayfair_dataset(filename, batch_size, output_dim, shuffle=True):
    batch_processor = dataset.GenericBatchProcessor(
            input_blocks=[dataset.blocks.TextPairGram(col=1)],
            label_blocks=[dataset.blocks.NumericalId(col=0, n_classes=output_dim, delimiter=",")],
            has_header=False,
            delimiter="\t"
        )

    dataloader = dataset.DatasetLoader(
        data_source=dataset.FileDataSource(filename=filename, batch_size=batch_size),
        batch_processor=batch_processor,
        shuffle=shuffle
    )
    data, labels = dataloader.load_in_memory()
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
        train_data, train_labels = load_wayfair_dataset(filename=config.train_dataset_path, batch_size=config.train_batch_size, output_dim=config.output_dim)
        test_data, test_labels = load_wayfair_dataset(filename=config.test_dataset_path, batch_size=config.test_batch_size, output_dim=config.output_dim, shuffle=False)

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


def get_train_and_eval_configs(benchmark_config):
    learning_rate = benchmark_config.learning_rate
    metrics = [benchmark_config.metric_type]

    train_config = bolt.TrainConfig(epochs=1, learning_rate=learning_rate).with_metrics(
        metrics
    )

    if hasattr(benchmark_config, "rehashing_factor"):
        train_config.with_reconstruct_hash_functions(benchmark_config.rehashing_factor)

    if hasattr(benchmark_config, "rebuild_hash_tables_factor"):
        train_config.with_rebuild_hash_tables(
            benchmark_config.rebuild_hash_tables_factor
        )

    eval_config = bolt.EvalConfig().with_metrics(metrics)

    return train_config, eval_config


def define_bolt_model(config: BoltBenchmarkConfig):
    def construct_criteo_dlrm_model(config):
        pass

    def get_loss_function(config):
        loss = config.loss_fn.lower()
        if loss == "categoricalcrossentropyloss":
            return bolt.nn.losses.CategoricalCrossEntropy()
        elif loss == "binarycrossentropyloss":
            return bolt.nn.losses.BinaryCrossEntropy()
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
    model = bolt.nn.Model(inputs=[input_node], output=output_node)
    model.compile(loss=loss_function, print_when_done=False)
    model.summary(detailed=True)

    return model


def get_mlflow_uri():
    load_dotenv()
    return os.getenv("MLFLOW_URI")

def compute_roc_auc():
    pass 


def launch_bolt_benchmark(config: BoltBenchmarkConfig, run_name: str) -> None:
    if not issubclass(config, BoltBenchmarkConfig):
        raise ValueError(
            f"The input config must be a bolt config. Given a config of type {config.__class__}"
        )

    mlflow_uri = get_mlflow_uri()

    model = define_bolt_model(config)
    train, train_y, test, test_y = load_datasets(config)
    
    epochs = config.num_epochs 
    train_config, eval_config = get_train_and_eval_configs(benchmark_config=config)

    for epoch in range(epochs):
        train_metrics = model.train(
            train_data=train,
            train_labels=train_y,
            train_config=train_config
        )
        print(f"train_metrics = {train_metrics}")

        predict_output = model.evaluate(
            test_data=test,
            test_labels=test_y,
            eval_config=eval_config
        )

        if config.compute_roc_auc:
            compute_roc_auc()

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
