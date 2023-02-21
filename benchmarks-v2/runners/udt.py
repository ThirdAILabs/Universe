from .runner import Runner
from thirdai import bolt, deployment
import json
from ..configs.udt_configs import UDTBenchmarkConfig


class UDTRunner(Runner):
    config_type = UDTBenchmarkConfig

    def run_benchmark(config: UDTBenchmarkConfig, path: str, mlflow_logger):
        if config.model_config is not None:
            deployment.dump_config(
                config=json.dumps(config.model_config),
                filename=config.model_config_path,
            )

        data_types = config.get_data_types()
        model = bolt.UniversalDeepTransformer(
            data_types=data_types,
            target=config.target,
            n_target_classes=config.n_target_classes,
            delimiter=config.delimiter,
            model_config=config.model_config_path,
        )

        model.train(
            path + config.train_file,
            epochs=config.num_epochs,
            learning_rate=config.learning_rate,
            callbacks=config.callbacks + [mlflow_logger] if mlflow_logger else [],
        )

        metrics = model.evaluate(
            path + config.test_file, metrics=config.metrics, return_metrics=True
        )

        if mlflow_logger:
            for k, v in metrics.items():
                mlflow_logger.log_additional_metric(key=k, value=v)
