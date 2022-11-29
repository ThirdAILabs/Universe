import argparse
import io
from contextlib import redirect_stdout

import mlflow
import udt_configs
from thirdai import bolt

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


def parse_eval_output(eval_output):
    metric = eval_output.split("|")[-3].strip()
    metric = metric.split(":")[-1][:-1]
    return metric


def run_benchmark(config):
    model = bolt.UniversalDeepTransformer(
        data_types=config.data_types,
        target=config.target,
        n_target_classes=config.n_target_classes,
        delimiter=config.delimiter,
    )

    model.train(
        config.train_file,
        epochs=config.num_epochs,
        learning_rate=config.learning_rate,
        metrics=["categorical_accuracy"],
        callbacks=[],
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
    start_mlflow_helper(
        experiment_name=args.config_name,
        run_name=args.run_name,
        dataset=args.config_name,
        model_name="udt",
    )
    log_machine_info()
    run_benchmark(config)


if __name__ == "__main__":
    main()
