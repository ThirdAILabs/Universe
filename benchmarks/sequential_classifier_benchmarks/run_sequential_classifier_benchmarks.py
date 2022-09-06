# TODO Return unordered map from strings to variant of string, int, and float. This is so we can effectively visualize parameters on MLFlow

import sys
import pathlib

from thirdai import bolt

sys.path.append(str(pathlib.Path(__file__).parent.resolve()) + "/..")
from utils import (
    config_get,
    config_get_or,
    find_full_filepath,
    log_metrics,
    log_params,
    start_experiment,
    start_mlflow,
)


def to_categorical_tuple(config, allow_delimiter=True):
    categorical_tuple = (
        config_get(config, "col_name"),
        config_get(config, "n_classes"),
    )
    if allow_delimiter and "delimiter" in config:
        categorical_tuple += (config_get(config, "delimiter"),)
    print(categorical_tuple)
    return categorical_tuple


def to_sequential_tuple(config):
    sequential_tuple = to_categorical_tuple(config)
    sequential_tuple += (config_get(config, "track_last_n"),)
    return sequential_tuple


def build_model(config):
    user_cfg = config_get(config, "user")
    target_cfg = config_get(config, "target")
    timestamp = config_get(config, "timestamp")
    static_text = config_get_or(config, "static_text", [])
    static_categorical_cfgs = config_get_or(config, "static_categorical", [])
    sequential_cfgs = config_get_or(config, "sequential", [])

    return bolt.SequentialClassifier(
        user=to_categorical_tuple(user_cfg, allow_delimiter=False),
        target=to_categorical_tuple(target_cfg),
        timestamp=timestamp,
        static_text=static_text,
        static_categorical=[
            to_categorical_tuple(cfg) for cfg in static_categorical_cfgs
        ],
        sequential=[to_sequential_tuple(cfg) for cfg in sequential_cfgs],
    )


def clean_dict_key(key):
    return key.replace('@', '_')


def prefix_dict_keys(dictionary, prefix):
    return {prefix + clean_dict_key(metric_name): value for (metric_name, value) in dictionary.items()}


def train_and_evaluate_one_epoch(model, config):
    train_data_config_path = config_get(config, "train_dataset_path")
    test_data_config_path = config_get(config, "test_dataset_path")

    train_data_full_path = find_full_filepath(train_data_config_path)
    test_data_full_path = find_full_filepath(test_data_config_path)
    prediction_file_path = config_get_or(config, "prediction_file_path", None)
    print_last_k = config_get_or(config, "predict_last_k", 1)

    learning_rate = config_get(config, "learning_rate")
    metrics = [
        "recall@1",
        "recall@5",
        "recall@10",
        "recall@25",
        "recall@50",
        "recall@100",
        "sampled@101_recall@1",
        "sampled@101_recall@10",
    ]

    logged_params = {"learning_rate": learning_rate}
    logged_metrics = {}

    train_metrics = model.train(
        train_data_full_path,
        epochs=1,
        learning_rate=learning_rate,
        metrics=metrics,
    )

    logged_params.update(model.hyperparameters())

    train_metrics = {key : val[-1] for (key, val) in train_metrics.items()}
    logged_metrics.update(prefix_dict_keys(train_metrics, "train_"))

    test_metrics = model.predict(
        test_data_full_path,
        metrics=metrics,
        output_file=prediction_file_path,
        print_last_k=print_last_k,
    )

    logged_metrics.update(prefix_dict_keys(test_metrics, "test_"))

    return logged_params, logged_metrics


def main():
    config, args = start_experiment(
        description="Trains and evaluates a bolt sequential classifier"
    )

    start_mlflow(config, args)

    model = build_model(config)
    epochs = config_get(config, "epochs")
    log_params({"epochs": epochs}, args)

    params, metrics = train_and_evaluate_one_epoch(model, config)

    log_params(params, args)
    log_metrics(metrics, args)

    for epoch in range(epochs - 1):
        _, metrics = train_and_evaluate_one_epoch(model, config)
        log_metrics(metrics, args)


if __name__ == "__main__":
    main()


# TODO train should return training metrics
# TODO test returns test metrics
# TODO write hyperparameters()
