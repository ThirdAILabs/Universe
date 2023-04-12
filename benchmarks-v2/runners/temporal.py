import json
import os
import pandas as pd
import time
import numpy as np

from thirdai import bolt, deployment

from ..configs.temporal_configs import UDTBenchmarkConfig
from .runner import Runner


class TemporalRunner(Runner):
    config_type = UDTBenchmarkConfig

    def run_benchmark(config: UDTBenchmarkConfig, path_prefix: str, mlflow_logger):
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

        # indexing train file so that train data user history is used for predictions
        model.evaluate(train_file)

        num_samples = 1000
        test_data = pd.read_csv(test_file, low_memory=False)
        sorted_idxs = np.sort(np.random.randint(0, len(test_data), size=num_samples))

        test_data_samples = []
        for _, row in test_data.iterrows():
            sample = dict(row)
            sample = {x: str(y) for x, y in sample.items()}
            test_data_samples.append(sample)

        test_data_sample = test_data.iloc[sorted_idxs]
        inference_samples = []
        for i, (_, row) in enumerate(test_data_sample.iterrows()):
            sample = dict(row)
            label = sample[config.target]
            del sample[config.target]
            sample = {x: str(y) for x, y in sample.items()}
            inference_samples.append((sample, label, sorted_idxs[i]))
                
        start_time = time.time()
        prev_idx = 0
        for sample, label, test_data_idx in inference_samples:
            if test_data_idx > prev_idx:
                model.index_batch(input_samples=test_data_samples[prev_idx:test_data_idx])
                prev_idx = test_data_idx
            model.predict(sample)
        end_time = time.time()
        time_per_predict = int(np.around(1000 * (end_time - start_time) / num_samples))

        print(f"average_predict_time = {time_per_predict}ms")
        if mlflow_logger:
            mlflow_logger.log_additional_metric(key="average_predict_time", value=time_per_predict)