import json
import os
import time

import numpy as np
import pandas as pd
from thirdai import bolt, deployment

from ..configs.query_reformulation_configs import *
from .runner import Runner


class QueryReformulationRunner(Runner):
    config_type = QueryReformulationBenchmarkConfig

    @classmethod
    def run_benchmark(
        cls, config: QueryReformulationBenchmarkConfig, path_prefix: str, mlflow_logger
    ):
        train_file = os.path.join(path_prefix, config.train_file)
        test_file = os.path.join(path_prefix, config.test_file)

        model = cls.create_model(config, path_prefix)

        model.train(train_file)

        for metric_name, metric_fn in config.additional_metric_fns.items():
            metric_value = metric_fn(model, test_file)

            print(f"{metric_name} = {metric_value}")
            if mlflow_logger:
                mlflow_logger.log_additional_metric(
                    key=f"val_{metric_name}", value=metric_value, step=0
                )

        average_predict_time_ms = cls.get_average_predict_time(
            model, test_file, config, path_prefix, num_samples=1000
        )

        print(f"average_predict_time_ms = {average_predict_time_ms}ms")
        if mlflow_logger:
            mlflow_logger.log_additional_metric(
                key="average_predict_time_ms", value=average_predict_time_ms
            )

    @staticmethod
    def create_model(config, path_prefix):
        if config.model_config is not None:
            model_config_path = config.config_name + "_model.config"
            deployment.dump_config(
                config=json.dumps(config.model_config),
                filename=model_config_path,
            )
        else:
            model_config_path = None

        model = bolt.UniversalDeepTransformer(
            data_types={
                config.source_column: bolt.types.text(),
                config.target_column: bolt.types.text(),
            },
            target=config.target_column,
            options={
                "dataset_size": config.dataset_size,
            },
        )

        if model_config_path:
            os.remove(model_config_path)

        return model

    @staticmethod
    def get_average_predict_time(
        model, test_file, config, path_prefix, num_samples=1000
    ):
        test_data = pd.read_csv(test_file, low_memory=False, delimiter=config.delimiter)
        test_data_sample = test_data.iloc[
            np.random.randint(0, len(test_data), size=num_samples)
        ]
        inference_samples = []
        for _, row in test_data_sample.iterrows():
            sample = dict(row)
            label = sample[config.target_column]
            sample = sample[config.source_column]
            inference_samples.append(({"phrase": sample}, label))

        start_time = time.time()
        for sample, label in inference_samples:
            model.predict(sample, top_k=5)
        end_time = time.time()
        average_predict_time_ms = float(
            np.around(1000 * (end_time - start_time) / num_samples, decimals=3)
        )
        return average_predict_time_ms
