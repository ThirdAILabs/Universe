import json
import os

import pandas as pd
import numpy as np
from thirdai import bolt, deployment
import time

from ..configs.udt_configs import UDTBenchmarkConfig
from ..configs.utils import AdditionalMetricCallback
from .runner import Runner



class UDTRunner(Runner):
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
        validation = bolt.Validation(
            test_file,
            metrics=config.metrics,
        )
        for callback in config.callbacks:
            if isinstance(callback, AdditionalMetricCallback):
                callback.set_test_file(test_file)
                callback.set_model(model)
                callback.set_mlflow_logger(mlflow_logger)

        if model_config_path:
            if os.path.join(path_prefix, config.model_config_path) != model_config_path:
                os.remove(model_config_path)

        # If the config has the neighbors type, we can assume a GNN backend
        contains_neighbors = any(
            [type(t) == bolt.types.neighbors for t in data_types.values()]
        )
        if contains_neighbors:
            test_file_dir = os.path.dirname(test_file)
            if not os.path.exists(os.path.join(test_file_dir, "gnn_index.csv")):
                df = pd.read_csv(test_file)
                df[config.target].values[:] = 0
                df.to_csv(os.path.join(test_file_dir, "gnn_index.csv"), index=False)
            model.index_nodes(os.path.join(test_file_dir, "gnn_index.csv"))

        if config.cold_start_num_epochs:
            model.cold_start(
                cold_start_train_file,
                epochs=config.cold_start_num_epochs,
                learning_rate=config.cold_start_learning_rate,
                strong_column_names=config.strong_column_names,
                weak_column_names=config.weak_column_names,
                validation=validation,
                callbacks=config.callbacks + [mlflow_logger] if mlflow_logger else [],
            )

        if config.num_epochs:
            model.train(
                train_file,
                epochs=config.num_epochs,
                learning_rate=config.learning_rate,
                validation=validation,
                callbacks=config.callbacks + [mlflow_logger] if mlflow_logger else [],
            )

        num_samples = 10000
        test_data = pd.read_csv(test_file, low_memory=False)
        test_data_sample = test_data.iloc[np.random.randint(0, len(test_data), size=num_samples)]
        inference_samples = []
        for _, row in test_data_sample.iterrows():
            sample = dict(row)
            label = sample[config.target]
            del sample[config.target]
            sample = {x: str(y) for x, y in sample.items()}
            inference_samples.append((sample, label))
                
        start_time = time.time()
        for sample, label in inference_samples:
            model.predict(sample)
        end_time = time.time()
        time_per_predict = int(np.around(1000 * (end_time - start_time) / num_samples))

        print(f"average_predict_time = {time_per_predict}ms")
        if mlflow_logger:
            mlflow_logger.log_additional_metric(key="average_predict_time", value=time_per_predict)