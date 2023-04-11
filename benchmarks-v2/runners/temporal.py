import json
import os

from thirdai import bolt, deployment

from ..configs.temporal_configs import TemporalBenchmarkConfig
from .runner import Runner


# This runner is temporary until https://github.com/ThirdAILabs/Universe/issues/1362 is addressed
class TemporalRunner(Runner):
    config_type = TemporalBenchmarkConfig

    def run_benchmark(config: TemporalBenchmarkConfig, path_prefix: str, mlflow_logger):
        train_file = (
            os.path.join(path_prefix, config.train_file)
            if config.train_file is not None
            else None
        )
        cold_start_train_file = (
            os.path.join(path_prefix, config.cold_start_train_file)
            if config.cold_start_train_file is not None
            else None
        )
        test_file = os.path.join(path_prefix, config.test_file)

        if config.model_config_path:
            model_config_path = os.path.join(path_prefix, config.model_config_path)
        elif config.model_config is not None:
            model_config_path = config.config_name + "_model.config"
            deployment.dump_config(
                config=json.dumps(config.model_config),
                filename=model_config_path,
            )
        else:
            model_config_path = None

        data_types = config.get_data_types(path_prefix)
        model = bolt.UniversalDeepTransformer(
            data_types=data_types,
            target=config.target,
            integer_target=config.integer_target,
            n_target_classes=config.n_target_classes,
            temporal_tracking_relationships=config.temporal_relationships,
            delimiter=config.delimiter,
            model_config=model_config_path,
            options=config.options,
        )

        for callback in config.callbacks:
            if isinstance(callback, AdditionalMetricCallback):
                callback.set_test_file(test_file)
                callback.set_model(model)
                callback.set_mlflow_logger(mlflow_logger)

        if model_config_path:
            if os.path.join(path_prefix, config.model_config_path) != model_config_path:
                os.remove(model_config_path)

        for epoch in range(config.num_epochs):
            model.train(
                train_file,
                epochs=1,
                learning_rate=config.learning_rate,
                callbacks=config.callbacks + [mlflow_logger] if mlflow_logger else [],
            )

            if len(config.metrics) > 0:
                metrics = model.evaluate(
                    test_file, metrics=config.metrics, return_metrics=True
                )

                if mlflow_logger:
                    for k, v in metrics.items():
                        mlflow_logger.log_additional_metric(key=k, value=v, step=epoch)

            model.reset_temporal_trackers()
