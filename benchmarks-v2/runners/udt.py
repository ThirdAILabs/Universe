import json
import os

from thirdai import bolt, deployment

from ..configs.udt_configs import UDTBenchmarkConfig
from .runner import Runner
from .utils import fix_mlflow_metric_name


class UDTRunner(Runner):
    config_type = UDTBenchmarkConfig

    def run_benchmark(config: UDTBenchmarkConfig, path: str, mlflow_logger):
        if config.model_config is not None:
            model_config_path = config.config_name + "_model.config"
            deployment.dump_config(
                config=json.dumps(config.model_config),
                filename=model_config_path,
            )
        else:
            model_config_path = None

        data_types = config.get_data_types()
        model = bolt.UniversalDeepTransformer(
            data_types=data_types,
            target=config.target,
            n_target_classes=config.n_target_classes,
            delimiter=config.delimiter,
            model_config=model_config_path,
        )

        if model_config_path:
            os.remove(model_config_path)

        for epoch in range(config.num_epochs):
            model.train(
                path + config.train_file,
                epochs=1,
                learning_rate=config.learning_rate,
                callbacks=config.callbacks + [mlflow_logger] if mlflow_logger else [],
            )

            if len(config.metrics) > 0:
                metrics = model.evaluate(
                    path + config.test_file, metrics=config.metrics, return_metrics=True
                )

                if mlflow_logger:
                    for k, v in metrics.items():
                        key = fix_mlflow_metric_name(k)
                        mlflow_logger.log_additional_metric(
                            key=key, value=v, step=epoch
                        )

            if config.additional_metric_fn:
                activations = model.evaluate(path + config.test_file)
                # The additional metric function allows for injecting a custom metric
                # that is not part of our builtin metrics in bolt.
                metric = config.additional_metric_fn(
                    activations=activations, mlflow_logger=mlflow_logger, step=epoch
                )
