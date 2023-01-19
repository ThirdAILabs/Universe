import os
import platform
import socket
import sys

import mlflow
import psutil
import thirdai
from thirdai import bolt as bolt_v1
from thirdai import bolt_v2 as bolt_v2
from thirdai import dataset


class BenchmarkConfig:
    input_dim = None

    hidden_dim = None
    hidden_sparsity = None

    output_dim = None
    output_sparsity = None
    output_sampling_config = None
    output_activation = None

    batch_size = None
    batches_per_rebuild_hash_tables = None
    batches_per_reconstruct_hash_functions = None
    learning_rates = None

    dataset_name = None


def train_bolt_v1(config: BenchmarkConfig, train_x, train_y, test_x, test_y):
    input_layer = bolt_v1.nn.Input(dim=config.input_dim)

    hidden = bolt_v1.nn.FullyConnected(
        dim=config.hidden_dim,
        sparsity=config.hidden_sparsity,
        activation="relu",
    )(input_layer)

    if config.output_sampling_config:
        output = bolt_v1.nn.FullyConnected(
            dim=config.output_dim,
            sparsity=config.output_sparsity,
            sampling_config=config.output_sampling_config,
            activation=config.output_activation,
        )(hidden)
    else:
        output = bolt_v1.nn.FullyConnected(
            dim=config.output_dim,
            sparsity=config.output_sparsity,
            activation=config.output_activation,
        )(hidden)

    model = bolt_v1.nn.Model(inputs=[input_layer], output=output)

    if config.output_activation == "softmax":
        loss = bolt_v1.nn.losses.CategoricalCrossEntropy()
    else:
        loss = bolt_v1.nn.losses.BinaryCrossEntropy()

    model.compile(loss, print_when_done=False)
    model.summary(detailed=True)

    eval_config = bolt_v1.EvalConfig().with_metrics(["categorical_accuracy"])

    for learning_rate in config.learning_rates:
        train_config = (
            bolt_v1.TrainConfig(epochs=1, learning_rate=learning_rate)
            .with_rebuild_hash_tables(
                config.batch_size * config.batches_per_rebuild_hash_tables
            )
            .with_reconstruct_hash_functions(
                config.batch_size * config.batches_per_reconstruct_hash_functions
            )
        )
        train_metrics = model.train([train_x], train_y, train_config)

        test_metrics = model.evaluate([test_x], test_y, eval_config)[0]

        metrics = {
            "epoch_time": train_metrics["epoch_times"][0],
            "val_time": test_metrics["test_time"],
            "categorical_accuracy": test_metrics["categorical_accuracy"],
        }
        mlflow.log_metrics(metrics)


def train_bolt_v2(config: BenchmarkConfig, train_x, train_y, test_x, test_y):
    input_layer = bolt_v2.nn.Input(dim=config.input_dim)

    hidden = bolt_v2.nn.FullyConnected(
        dim=config.hidden_dim,
        input_dim=config.input_dim,
        sparsity=config.hidden_sparsity,
        activation="relu",
        rebuild_hash_tables=config.batches_per_rebuild_hash_tables,
        reconstruct_hash_functions=config.batches_per_reconstruct_hash_functions,
    )(input_layer)

    output = bolt_v2.nn.FullyConnected(
        dim=config.output_dim,
        input_dim=config.hidden_dim,
        sparsity=config.output_sparsity,
        sampling_config=config.output_sampling_config,
        activation=config.output_activation,
        rebuild_hash_tables=config.batches_per_rebuild_hash_tables,
        reconstruct_hash_functions=config.batches_per_reconstruct_hash_functions,
    )(hidden)

    if config.output_activation == "softmax":
        loss = bolt_v2.nn.losses.CategoricalCrossEntropy(output)
    else:
        loss = bolt_v2.nn.losses.BinaryCrossEntropy(output)

    model = bolt_v2.nn.Model(inputs=[input_layer], outputs=[output], losses=[loss])

    model.summary()

    trainer = bolt_v2.train.Trainer(model)

    for learning_rate in config.learning_rates:
        history = trainer.train(
            train_data=(train_x, train_y),
            epochs=1,
            learning_rate=learning_rate,
            train_metrics={},
            validation_data=(test_x, test_y),
            validation_metrics={
                output.name(): [bolt_v2.train.metrics.CategoricalAccuracy()],
            },
            steps_per_validation=None,
            callbacks=[],
        )

        metrics = {
            "epoch_time": history["all"]["epoch_times"][0],
            "val_time": history["all"]["val_times"][0],
            "categorical_accuracy": history[output.name()]["val_categorical_accuracy"][0],
        }
        mlflow.log_metrics(metrics)


def init_mlflow(dataset_name, bolt_version, suffix):
    mlflow.set_tracking_uri(
        sys.argv[1]
    )
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
    mlflow.set_experiment("Bolt V2 Benchmarking")
    mlflow.start_run(
        run_name=f"{bolt_version}_{dataset_name}_{suffix}",
        tags={"dataset": dataset_name, "bolt_version": bolt_version},
    )
    mlflow.log_params(machine_info)
    mlflow.log_param("thirdai_version", thirdai.__version__)
    mlflow.log_artifact(__file__)


def run_experiment(config: BenchmarkConfig, data_load_fn, suffix):
    train_x, train_y, test_x, test_y = data_load_fn()

    init_mlflow(dataset_name=config.dataset_name, bolt_version="bolt_v1", suffix=suffix)
    train_bolt_v1(config, train_x, train_y, test_x, test_y)
    mlflow.end_run()

    init_mlflow(dataset_name=config.dataset_name, bolt_version="bolt_v2", suffix=suffix)
    train_bolt_v2(config, train_x, train_y, test_x, test_y)
    mlflow.end_run()


class Amazon670Config:
    input_dim = 135909

    hidden_dim = 256
    hidden_sparsity = 1.0

    output_dim = 670091
    output_sparsity = 0.005
    output_sampling_config = None
    output_activation = "softmax"

    batch_size = 256
    batches_per_rebuild_hash_tables = 25
    batches_per_reconstruct_hash_functions = 500
    learning_rates = [0.0001] * 5

    dataset_name = "amazon_670"


def amazon_670_data():
    train_file = "/share/data/amazon-670k/train_shuffled_noHeader.txt"
    test_file = "/share/data/amazon-670k/test_shuffled_noHeader_sampled.txt"
    train_x, train_y = dataset.load_bolt_svm_dataset(
        train_file, Amazon670Config.batch_size
    )
    test_x, test_y = dataset.load_bolt_svm_dataset(
        test_file, Amazon670Config.batch_size
    )
    return train_x, train_y, test_x, test_y


class AmazonPolarityConfig:
    input_dim = 100000

    hidden_dim = 10000
    hidden_sparsity = 0.005

    output_dim = 2
    output_sparsity = 1.0
    output_sampling_config = None
    output_activation = "softmax"

    batch_size = 256
    batches_per_rebuild_hash_tables = 25
    batches_per_reconstruct_hash_functions = 500
    learning_rates = [0.0001] * 5

    dataset_name = "amazon_polarity"


def amazon_polarity_data():
    train_file = "/share/data/amazon_polarity/svm_train.txt"
    test_file = "/share/data/amazon_polarity/svm_test.txt"
    train_x, train_y = dataset.load_bolt_svm_dataset(
        train_file, Amazon670Config.batch_size
    )
    test_x, test_y = dataset.load_bolt_svm_dataset(
        test_file, Amazon670Config.batch_size
    )
    return train_x, train_y, test_x, test_y


class WayfairConfig:
    input_dim = 100000

    hidden_dim = 1024
    hidden_sparsity = 1.0

    output_dim = 931
    output_sparsity = 0.1
    output_sampling_config = bolt_v1.nn.DWTASamplingConfig(
        num_tables=64,
        hashes_per_table=4,
        reservoir_size=64,
    )
    output_activation = "sigmoid"

    batch_size = 2048
    batches_per_rebuild_hash_tables = 5
    batches_per_reconstruct_hash_functions = 25
    learning_rates = [0.001] * 3 + [0.0001] * 2

    dataset_name = "wayfair"


def load_wayfair_data(filename, shuffle):
    processor = dataset.GenericBatchProcessor(
        input_blocks=[dataset.blocks.TextPairGram(col=1)],
        label_blocks=[dataset.blocks.NumericalId(col=0, n_classes=931, delimiter=",")],
        has_header=False,
        delimiter="\t",
    )

    loader = dataset.DatasetLoader(
        data_source=dataset.FileDataSource(filename=filename, batch_size=WayfairConfig.batch_size),
        batch_processor=processor,
        shuffle=shuffle,
    )

    return loader.load_in_memory()


def wayfair_data():
    train_file = "/share/data/wayfair_2/train_bert_auto_clf.txt"
    test_file = "/share/data/wayfair_2/dev_bert_auto_clf.txt"

    train_x, train_y = load_wayfair_data(train_file, True)
    test_x, test_y = load_wayfair_data(test_file, False)
    return train_x[0], train_y, test_x[0], test_y


class MnistConfig:
    input_dim = 784

    hidden_dim = 20000
    hidden_sparsity = 0.01

    output_dim = 10
    output_sparsity = 1.0
    output_sampling_config = None
    output_activation = "softmax"

    batch_size = 250
    batches_per_rebuild_hash_tables = 10
    batches_per_reconstruct_hash_functions = 40
    learning_rates = [0.0001]


def mnist_data():
    train_x, train_y = dataset.load_bolt_svm_dataset(
        "/Users/nmeisburger/ThirdAI/data/mnist/mnist", MnistConfig.batch_size
    )
    test_x, test_y = dataset.load_bolt_svm_dataset(
        "/Users/nmeisburger/ThirdAI/data/mnist/mnist.t", MnistConfig.batch_size
    )
    return train_x, train_y, test_x, test_y


if __name__ == "__main__":
    run_experiment(Amazon670Config, amazon_670_data, sys.argv[2])
    run_experiment(AmazonPolarityConfig, amazon_polarity_data, sys.argv[2])
    run_experiment(WayfairConfig, wayfair_data, sys.argv[2])
