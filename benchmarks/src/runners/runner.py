import json
import os
from abc import ABC, abstractmethod

from thirdai import bolt, deployment


class Runner(ABC):
    @property
    @staticmethod
    @abstractmethod
    def config_type():
        """
        Returns the type of config supported by the given runner.
        """
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def run_benchmark(cls, config, path_prefix, mlflow_logger=None):
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def get_average_predict_time(
        model, test_file, config, path_prefix, num_samples=10000
    ):
        raise NotImplementedError

    @staticmethod
    def create_model(config, path_prefix):
        data_types = config.get_data_types(path_prefix)
        model = bolt.UniversalDeepTransformer(
            data_types=data_types,
            target=config.target,
            integer_target=config.integer_target,
            n_target_classes=config.n_target_classes,
            temporal_tracking_relationships=config.temporal_relationships,
            delimiter=config.delimiter,
            options=config.options,
        )

        if config.custom_model:
            model._set_model(config.custom_model())

        return model
