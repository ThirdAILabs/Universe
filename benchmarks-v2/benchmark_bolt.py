import argparse
from typing import Union

import numpy as np
from configs import bolt_configs
from thirdai import bolt, dataset
from thirdai.experimental import MlflowCallback


def get_mlflow_callback(config, run_name, mlflow_uri):
    experiment_name = config.experiment_name
    dataset_name = config.dataset_name

    mlflow_callback = MlflowCallback(
        tracking_uri=mlflow_uri,
        experiment_name=experiment_name,
        run_name=run_name,
        dataset_name=dataset_name,
        experiment_args={},
    )
    return mlflow_callback


def get_train_and_eval_configs(benchmark_config, callbacks=None):

    learning_rate = benchmark_config.learning_rate
    metrics = [benchmark_config.metric_type]

    train_config = bolt.TrainConfig(epochs=1, learning_rate=learning_rate).with_metrics(
        metrics
    )
    if callbacks is not None:
        train_config.with_callbacks(callbacks)

    if hasattr(benchmark_config, "reconstruct_hash_functions"):
        train_config.with_reconstruct_hash_functions(
            benchmark_config.reconstruct_hash_functions
        )

    if hasattr(benchmark_config, "rebuild_hash_tables"):
        train_config.with_rebuild_hash_tables(benchmark_config.rebuild_hash_tables)

    eval_config = bolt.EvalConfig().with_metrics(metrics)
    if benchmark_config.compute_roc_auc == True:
        eval_config.return_activations()

    return train_config, eval_config


def define_dlrm_model(config):

    input_node = bolt.nn.Input(dim=config.input_dim)
    token_input = bolt.nn.TokenInput(**config.token_input)

    first_hidden_node = bolt.nn.FullyConnected(**config.first_hidden_node)(input_node)
    second_hidden_node = bolt.nn.FullyConnected(**config.second_hidden_node)(
        first_hidden_node
    )

    embedding_node = bolt.nn.FullyConnected(**config.embedding_node)(token_input)
    concat_node = bolt.nn.Concatenate()[second_hidden_node, embedding_node]

    third_hidden_node = bolt.nn.FullyConnected(**config.third_hidden_node)(concat_node)
    output_node = bolt.nn.FullyConnected(**config.output_node)(third_hidden_node)

    model = bolt.nn.Model(inputs=[input_node, token_input], output=output_node)
    model.compile(
        loss=bolt.nn.losses.get_loss_function(name=config.loss_fn),
        print_when_done=False,
    )
    model.summary(detailed=True)
    return model


def define_bolt_model(config):
    input_node = bolt.nn.Input(dim=config.input_dim)
    hidden_node = bolt.nn.FullyConnected(**config.hidden_node)(input_node)
    output_node = bolt.nn.FullyConnected(**config.output_node)(hidden_node)

    model = bolt.nn.Model(inputs=[input_node], output=output_node)
    model.compile(
        loss=bolt.nn.losses.get_loss_function(name=config.loss_fn),
        print_when_done=False,
    )
    model.summary(detailed=True)

    return model


def compute_roc_auc(predict_output, test_labels_path, mlflow_callback=None):
    with open(test_labels_path) as file:
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


def run_dlrm_benchmark(config, callbacks):
    model = define_dlrm_model()
    train_set, train_labels, test_set, test_labels = config.load_datasets()

    train_config, eval_config = get_train_and_eval_configs(
        benchmark_config=config, callbacks=callbacks
    )

    for _ in range(config.num_epochs):
        train_metrics = model.train(
            train_data=train_set, train_labels=train_labels, train_config=train_config
        )
        for k, v in train_metrics.items():
            mlflow_callback.log_additional_metric(key=k, value=v[0])

        predict_output = model.evaluate(
            test_data=test_set, test_labels=test_labels, eval_config=eval_config
        )
        mlflow_callback.log_additional_metric(
            key=predict_output[0][0], value=predict_output[0][1]
        )
        compute_roc_auc(
            predict_output=predict_output,
            test_labels_path=config.test_dataset_path,
            mlflow_callback=mlflow_callback,
        )


def run_fully_connected_benchmark(config, callbacks):
    model = define_bolt_model(config)
    train_set, train_labels, test_set, test_labels = config.load_datasets()

    train_config, eval_config = get_train_and_eval_configs(
        benchmark_config=config, callbacks=callbacks
    )

    for _ in range(config.num_epochs):
        train_metrics = model.train(
            train_data=train_set, train_labels=train_labels, train_config=train_config
        )

        for k, v in train_metrics.items():
            mlflow_callback.log_additional_metric(key=k, value=v[0])

        predict_output = model.evaluate(
            test_data=test_set, test_labels=test_labels, eval_config=eval_config
        )
        mlflow_callback.log_additional_metric(
            key=predict_output[0][0], value=predict_output[0][1]
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark a dataset with Bolt")

    parser.add_argument(
        "--runner",
        required=True,
        help="Specify the runner type for Bolt benchmarks. Options include 'fully_connected' and 'dlrm'",
    )
    parser.add_argument(
        "--mlflow_uri", required=True, help="MLflow URI to log metrics and artifacts."
    )
    parser.add_argument(
        "--run_name", required=True, help="The job name to track in MLflow"
    )
    parser.add_argument(
        "--config_name",
        default="",
        required=True,
        help="The python class name of the Bolt benchmark config",
    )

    args = parser.parse_args()
    config = getattr(bolt_configs, args.config_name)

    runner = args.runner
    mlflow_callback = get_mlflow_callback(
        config=config, run_name=args.run_name, mlflow_uri=args.mlflow_uri
    )
    callbacks = [mlflow_callback]
    callbacks.extend(config.callbacks)

    if runner.lower() == "dlrm":
        run_dlrm_benchmark(config=config, callbacks=callbacks)
    elif runner.lower() == "fully_connected":
        run_fully_connected_benchmark(config=config, callbacks=callbacks)
    else:
        raise ValueError(f"Invalid runner: {runner}")
