import os

from thirdai import bolt

from ...configs.temporal_configs import *
from ..temporal import TemporalRunner
from .utils import OLD_MODEL_PATH, get_filtered_versions


class BackwardCompatibilityTemporalRunner(TemporalRunner):
    config_type = TemporalBenchmarkConfig

    @staticmethod
    def create_model(config, path_prefix):
        print(BackwardCompatibilityTemporalRunner.old_model_path)
        model = bolt.UniversalDeepTransformer.load(
            BackwardCompatibilityTemporalRunner.old_model_path
        )

        return model

    @classmethod
    def run_benchmark(
        cls, config: TemporalBenchmarkConfig, path_prefix: str, mlflow_logger
    ):
        if config.num_epochs:
            config.num_epochs = 1

        filtered_versions = get_filtered_versions()

        for filtered_version in filtered_versions:
            print(filtered_version)
            formatted_version = filtered_version.replace(".", "_")
            BackwardCompatibilityTemporalRunner.old_model_path = os.path.join(
                OLD_MODEL_PATH, f"{formatted_version}_{config.config_name}.model"
            )
            TemporalRunner.run_benchmark.__func__(
                BackwardCompatibilityTemporalRunner, config, path_prefix, mlflow_logger
            )
