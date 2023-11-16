import os
import time

import numpy as np
import pandas as pd

from ..configs.temporal_configs import *
from ..configs.utils import AdditionalMetricCallback
from .runner import Runner


class TemporalRunner(Runner):
    config_type = TemporalBenchmarkConfig

    @classmethod
    def run_benchmark(
        cls, config: TemporalBenchmarkConfig, path_prefix: str, mlflow_logger
    ):
        train_file = (
            os.path.join(path_prefix, config.train_file)
            if config.train_file is not None
            else None
        )
        test_file = os.path.join(path_prefix, config.test_file)

        model = cls.create_model(config, path_prefix)

        for callback in config.callbacks:
            if isinstance(callback, AdditionalMetricCallback):
                callback.set_test_file(test_file)
                callback.set_model(model)
                callback.set_mlflow_logger(mlflow_logger)

        for epoch in range(config.num_epochs):
            model.train(
                train_file,
                epochs=1,
                learning_rate=config.learning_rate,
                max_in_memory_batches=config.max_in_memory_batches,
                callbacks=config.callbacks + ([mlflow_logger] if mlflow_logger else []),
            )

            if len(config.metrics) > 0:
                metrics = model.evaluate(test_file, metrics=config.metrics)

                if mlflow_logger:
                    for k, v in metrics.items():
                        mlflow_logger.log_additional_metric(
                            key=k, value=v[-1], step=epoch
                        )

            model.reset_temporal_trackers()

        # indexing train file so that train data user history is used for predictions
        train_data = pd.read_csv(
            train_file, low_memory=False, delimiter=config.delimiter
        )
        for _, row in train_data.iterrows():
            sample = dict(row)
            sample = {x: str(y) for x, y in sample.items()}
            model.index(sample)

        del train_data

        average_predict_time_ms = cls.get_average_predict_time(
            model, test_file, config, path_prefix, num_samples=1000
        )

        print(f"average_predict_time_ms = {average_predict_time_ms}ms")
        if mlflow_logger:
            mlflow_logger.log_additional_metric(
                key="average_predict_time_ms", value=average_predict_time_ms
            )

    @staticmethod
    def get_average_predict_time(
        model, test_file, config, path_prefix, num_samples=1000
    ):
        test_data = pd.read_csv(test_file, low_memory=False, delimiter=config.delimiter)
        sorted_idxs = np.sort(np.random.randint(0, len(test_data), size=num_samples))

        test_data_samples = []
        for _, row in test_data.iterrows():
            sample = dict(row)
            sample = {x: str(y) for x, y in sample.items()}
            test_data_samples.append(sample)

        test_data_sample = test_data.iloc[sorted_idxs]
        del test_data
        inference_samples = []
        sample_col_names = config.get_data_types(path_prefix).keys()
        for i, (_, row) in enumerate(test_data_sample.iterrows()):
            sample = dict(row)
            label = sample[config.target]
            del sample[config.target]
            sample = {x: str(y) for x, y in sample.items() if x in sample_col_names}
            inference_samples.append((sample, label, sorted_idxs[i]))

        start_time = time.time()
        prev_idx = 0
        for sample, label, test_data_idx in inference_samples:
            if test_data_idx > prev_idx:
                model.index_batch(
                    input_samples=test_data_samples[prev_idx:test_data_idx]
                )
                prev_idx = test_data_idx
            model.predict(sample)
        end_time = time.time()
        average_predict_time_ms = float(
            np.around(1000 * (end_time - start_time) / num_samples, decimals=3)
        )
        return average_predict_time_ms
