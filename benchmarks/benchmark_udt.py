import argparse
import os

import mlflow
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


def run_benchmark(config, run_name):
    model = bolt.UniversalDeepTransformer(
        data_types=config.data_types,
        target=config.target,
        n_target_classes=config.n_target_classes,
        delimiter=config.delimiter,
    )

    mlflow_uri = get_mlflow_uri()

    mlflowcallback = MlflowCallback(
        mlflow_uri,
        config.experiment_name,
        run_name,
        config.dataset_name,
        {},
    )

    model.train(
        config.train_file,
        epochs=config.num_epochs,
        learning_rate=config.learning_rate,
        metrics=["categorical_accuracy"],
        callbacks=[mlflowcallback],
    )

    f = io.StringIO()
    with redirect_stdout(f):
        model.evaluate(config.test_file, metrics=["categorical_accuracy"])

    eval_output = f.getvalue()
    eval_accuracy = parse_eval_output(eval_output)
    mlflow.log_metric("categorical_accuracy", eval_accuracy)

def main():
    args = parse_args()
    config = getattr(udt_configs, args.config_name)
    run_benchmark(config, args.run_name)


if __name__ == "__main__":
    main()
