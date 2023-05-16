from ...configs.cold_start_configs import *
from ...configs.graph_configs import *
from ...configs.mach_configs import *
from ...configs.udt_configs import *
from ..udt import UDTRunner
from thirdai import bolt
import os
from .utils import get_package_versions, OLD_MODEL_PATH


class BackwardCompatibilityUDTRunner(UDTRunner):
    config_type = UDTBenchmarkConfig
    old_model_path = ""

    @staticmethod
    def create_model(config, path_prefix):
        print(BackwardCompatibilityUDTRunner.old_model_path)
        model = bolt.UniversalDeepTransformer.load(BackwardCompatibilityUDTRunner.old_model_path)

        return model

    @classmethod
    def run_benchmark(cls, config: UDTBenchmarkConfig, path_prefix: str, mlflow_logger):
        if config.num_epochs:
            config.num_epochs = 1
        if config.cold_start_num_epochs:
            config.cold_start_num_epochs = 1

        with open("thirdai.version") as version_file:
            full_version = version_file.read().strip()
            minor_version = ".".join(full_version.split(".")[:-1]) + "."

        filtered_versions = [version for version in get_package_versions("thirdai") if version[:len(minor_version)] == minor_version]

        for filtered_version in filtered_versions:
            BackwardCompatibilityUDTRunner.old_model_path = os.path.join(OLD_MODEL_PATH, f"{filtered_version}_{config.config_name}.model")
            UDTRunner.run_benchmark.__func__(BackwardCompatibilityUDTRunner, config, path_prefix, mlflow_logger)

        
