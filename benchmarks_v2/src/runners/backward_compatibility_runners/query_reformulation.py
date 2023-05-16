from ...configs.query_reformulation_configs import *
from ..query_reformulation import QueryReformulationRunner
from thirdai import bolt
import os
from .utils import get_package_versions, OLD_MODEL_PATH


class BackwardCompatibilityQueryReformulationRunner(QueryReformulationRunner):
    config_type = QueryReformulationBenchmarkConfig

    @staticmethod
    def create_model(config, path_prefix):
        print(BackwardCompatibilityQueryReformulationRunner.old_model_path)
        model = bolt.UniversalDeepTransformer.load(BackwardCompatibilityQueryReformulationRunner.old_model_path)

        return model

    @classmethod
    def run_benchmark(
        cls, config: QueryReformulationBenchmarkConfig, path_prefix: str, mlflow_logger
    ):
        config.dataset_size = "small"

        with open("thirdai.version") as version_file:
            full_version = version_file.read().strip()
            minor_version = ".".join(full_version.split(".")[:-1]) + "."

        filtered_versions = [version for version in get_package_versions("thirdai") if version[:len(minor_version)] == minor_version]

        for filtered_version in filtered_versions:
            BackwardCompatibilityQueryReformulationRunner.old_model_path = os.path.join(OLD_MODEL_PATH, f"{filtered_version}_{config.config_name}.model")
            QueryReformulationRunner.run_benchmark.__func__(BackwardCompatibilityQueryReformulationRunner, config, path_prefix, mlflow_logger)

