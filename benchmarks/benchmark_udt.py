import argparse
import io
import os
from contextlib import redirect_stdout

import mlflow
import numpy as np
import toml
import udt_configs
from thirdai import bolt
from thirdai.experimental import MlflowCallback

from utils import log_machine_info, start_mlflow_helper


def parse_args():
    parser = argparse.ArgumentParser(description="Benchmark a dataset with UDT")
    parser.add_argument(
        "--run_name", default="", help="The job name to track in mlflow"
    )
    parser.add_argument(
        "--config_name",
        help="The python class name of the UDT benchmark config",
    )

    args = parser.parse_args()
    return args


def get_mlflow_uri():
    file_dir = os.path.dirname(os.path.abspath(__file__))
    file_name = os.path.join(file_dir, "config.toml")
    with open(file_name) as f:
        parsed_config = toml.load(f)
        return parsed_config["tracking"]["uri"]


def parse_metric(stdout_handler, metric_type):
    import re

    output = stdout_handler.getvalue()
    metric = re.search(f"{metric_type}\s*:\s*0.[0-9][0-9][0-9]", output).group(0)
    metric = metric.split(":")[-1]
    return metric


def evaluate_over_file(test_file, model, metric_type):
    stdout_handler = io.StringIO()
    with redirect_stdout(stdout_handler):
        model.evaluate(test_file, metrics=[metric_type])
        metric = parse_metric(stdout_handler, metric_type)
        return metric


def evaluate_over_part_files(test_files, model, metric_type):
    """
    Evaluate over a list of files and then compute an
    overall average metric
    """
    totals = []
    num_lines_list = [sum(1 for line in open(f)) for f in test_files]
    for test_file, num_lines in zip(test_files, num_lines_list):
        stdout_handler = io.StringIO()
        with redirect_stdout(stdout_handler):
            model.evaluate(test_file, metrics=[metric_type])
            metric = parse_metric(stdout_handler, metric_type)
            weighted_total = float(metric) * num_lines
            totals.append(weighted_total)

    return sum(totals) / sum(num_lines_list)


def run_benchmark(config, run_name):
    if config.model_config is not None:
        print(config.model_config_path)
        config.model_config.save(config.model_config_path)

    model = bolt.UniversalDeepTransformer(
        data_types=config.data_types,
        target=config.target,
        n_target_classes=config.n_target_classes,
        delimiter=config.delimiter,
        model_config=config.model_config_path,
    )

    mlflow_uri = get_mlflow_uri()
    mlflow_callback = MlflowCallback(
        mlflow_uri,
        config.experiment_name,
        run_name,
        config.dataset_name,
        {},
    )

    callbacks = [mlflow_callback]
    callbacks.extend(config.callbacks)

    model.train(
        config.train_file,
        epochs=config.num_epochs,
        learning_rate=config.learning_rate,
        metrics=[config.metric_type],
        callbacks=callbacks,
    )

    if isinstance(config.test_file, list):
        test_metric = evaluate_over_part_files(
            config.test_file, model, config.metric_type
        )
    else:
        test_metric = evaluate_over_file(config.test_file, model, config.metric_type)

    mlflow_callback.log_additional_metric(f"test_{config.metric_type}", test_metric)


def main():
    args = parse_args()
    config = getattr(udt_configs, args.config_name)
    run_benchmark(config, args.run_name)


if __name__ == "__main__":
    main()
