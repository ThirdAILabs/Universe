import os
import time

import numpy as np
import pandas as pd
from thirdai import bolt

from ..configs.cold_start_configs import *
from ..configs.graph_configs import *
from ..configs.mach_configs import *
from ..configs.udt_configs import *
from ..configs.utils import AdditionalMetricCallback
from .runner import Runner


class UDTRunner(Runner):
    config_type = UDTBenchmarkConfig

    @classmethod
    def run_benchmark(cls, config: UDTBenchmarkConfig, path_prefix: str, mlflow_logger):
        train_file, cold_start_train_file, test_file = cls.get_datasets(
            config, path_prefix
        )

        model = cls.create_model(config, path_prefix)

        validation = (
            bolt.Validation(
                test_file,
                metrics=config.metrics,
            )
            if config.metrics
            else None
        )

        for callback in config.callbacks:
            if isinstance(callback, AdditionalMetricCallback):
                callback.set_test_file(test_file)
                callback.set_model(model)
                callback.set_mlflow_logger(mlflow_logger)

        has_gnn_backend = any(
            [
                type(t) == bolt.types.neighbors
                for t in config.get_data_types(path_prefix).values()
            ]
        )
        if has_gnn_backend:
            test_file_dir = os.path.dirname(test_file)
            if not os.path.exists(os.path.join(test_file_dir, "gnn_index.csv")):
                df = pd.read_csv(test_file)
                df[config.target].values[:] = 0
                df.to_csv(os.path.join(test_file_dir, "gnn_index.csv"), index=False)
            model.index_nodes(os.path.join(test_file_dir, "gnn_index.csv"))

        if cold_start_train_file is not None:
            model.cold_start(
                cold_start_train_file,
                epochs=config.cold_start_num_epochs,
                learning_rate=config.cold_start_learning_rate,
                strong_column_names=config.strong_column_names,
                weak_column_names=config.weak_column_names,
                validation=validation,
                callbacks=config.callbacks + ([mlflow_logger] if mlflow_logger else []),
            )

        if train_file is not None:
            model.train(
                train_file,
                epochs=config.num_epochs,
                learning_rate=config.learning_rate,
                validation=validation,
                max_in_memory_batches=config.max_in_memory_batches,
                callbacks=config.callbacks + ([mlflow_logger] if mlflow_logger else []),
            )

        average_predict_time_ms = cls.get_average_predict_time(
            model, test_file, config, path_prefix, 1000
        )

        print(f"average_predict_time_ms = {average_predict_time_ms}ms")
        if mlflow_logger:
            mlflow_logger.log_additional_metric(
                key="average_predict_time_ms", value=average_predict_time_ms
            )

    @staticmethod
    def get_datasets(config, path_prefix):
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
        return train_file, cold_start_train_file, test_file

    @staticmethod
    def get_average_predict_time(
        model, test_file, config, path_prefix, num_samples=1000
    ):
        test_data = pd.read_csv(test_file, low_memory=False, delimiter=config.delimiter)
        test_data_sample = test_data.iloc[
            np.random.randint(0, len(test_data), size=num_samples)
        ]
        inference_samples = []
        sample_col_names = config.get_data_types(path_prefix).keys()
        for _, row in test_data_sample.iterrows():
            sample = dict(row)
            label = sample[config.target]
            del sample[config.target]
            sample = {x: str(y) for x, y in sample.items() if x in sample_col_names}
            inference_samples.append((sample, label))

        start_time = time.time()
        for sample, label in inference_samples:
            model.predict(sample)
        end_time = time.time()
        average_predict_time_ms = float(
            np.around(1000 * (end_time - start_time) / num_samples, decimals=3)
        )
        return average_predict_time_ms
