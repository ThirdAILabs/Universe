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
        "--config_name", help="The python class name of the UDT benchmark config",
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
        delimiter='\t', #config.delimiter,
        model_config=config.model_config_path,
    )

    mlflow_uri = get_mlflow_uri()

    callbacks = [
        MlflowCallback(
            mlflow_uri, config.experiment_name, run_name, config.dataset_name, {},
        )
    ]
    # callbacks.extend(config.callbacks)

    model.train(
        config.train_file,
        epochs=config.num_epochs,
        learning_rate=config.learning_rate,
        metrics=[config.metric_type],
        callbacks=callbacks,
    )

    model.evaluate(config.test_file, metrics=[config.metric_type])


def main():
    args = parse_args()
    config = getattr(udt_configs, args.config_name)
    run_benchmark(config, args.run_name)


if __name__ == "__main__":
    main()
