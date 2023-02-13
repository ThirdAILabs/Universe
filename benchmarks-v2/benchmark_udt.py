import argparse
import json

import udt_configs
from configs import udt_configs
from thirdai import bolt, deployment
from thirdai.experimental import MlflowCallback


def run_udt_benchmark(config, run_name, mlflow_uri):
    if config.model_config is not None:
        deployment.dump_config(
            config=json.dumps(config.model_config), filename=config.model_config_path
        )

    data_types = config.data_types
    model = bolt.UniversalDeepTransformer(
        data_types=data_types,
        target=config.target,
        n_target_classes=config.n_target_classes,
        delimiter=config.delimiter,
        model_config=config.model_config_path,
    )

    callbacks = [
        MlflowCallback(
            tracking_uri=mlflow_uri,
            experiment_name=config.experiment_name,
            run_name=run_name,
            dataset_name=config.dataset_name,
            experiment_args={},
        )
    ]
    callbacks.extend(config.callbacks)

    model.train(
        config.train_file,
        epochs=config.num_epochs,
        learning_rate=config.learning_rate,
        metrics=[config.metric_type],
        callbacks=callbacks,
    )

    model.evaluate(config.test_file, metrics=[config.metric_type])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark a dataset with UDT")
    parser.add_argument(
        "--runner",
        default="udt",
        required=True,
        help="Specify the runner type for UDT benchmarks. It should always be 'udt'",
    )
    parser.add_argument(
        "--mlflow_uri", required=True, help="MLflow URI to log metrics and artifacts."
    )
    parser.add_argument(
        "--run_name", default="", required=True, help="The job name to track in MLflow"
    )
    parser.add_argument(
        "--config_name",
        default="",
        required=True,
        help="The python class name of the UDT benchmark config",
    )

    args = parser.parse_args()
    config = getattr(udt_configs, args.config_name)

    if args.runner.lower() != "udt":
        raise ValueError(f"Invalid runner type {args.runner} for UDT benchmark.")

    run_udt_benchmark(config=config, run_name=args.run_name, mlflow_uri=args.mlflow_uri)
