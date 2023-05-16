from ...configs.temporal_configs import *
from ..temporal import TemporalRunner
from thirdai import bolt
import os
from .utils import get_package_versions, OLD_MODEL_PATH


class BackwardCompatibilityTemporalRunner(TemporalRunner):
    config_type = TemporalBenchmarkConfig

    @staticmethod
    def create_model(config, path_prefix):
        print(BackwardCompatibilityTemporalRunner.old_model_path)
        model = bolt.UniversalDeepTransformer.load(BackwardCompatibilityTemporalRunner.old_model_path)

        return model

    @classmethod
    def run_benchmark(
        cls, config: TemporalBenchmarkConfig, path_prefix: str, mlflow_logger
    ):
        if config.num_epochs:
            config.num_epochs = 1

        with open("thirdai.version") as version_file:
            full_version = version_file.read().strip()
            minor_version = ".".join(full_version.split(".")[:-1]) + "."

        filtered_versions = [version for version in get_package_versions("thirdai") if version[:len(minor_version)] == minor_version]

        for filtered_version in filtered_versions:
            BackwardCompatibilityTemporalRunner.old_model_path = os.path.join(OLD_MODEL_PATH, f"{filtered_version}_{config.config_name}.model")
            TemporalRunner.run_benchmark.__func__(BackwardCompatibilityTemporalRunner, config, path_prefix, mlflow_logger)
