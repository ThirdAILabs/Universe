import os

from thirdai import bolt

from ...configs.query_reformulation_configs import *
from ..query_reformulation import QueryReformulationRunner
from .utils import OLD_MODEL_PATH, get_filtered_versions


class BackwardCompatibilityQueryReformulationRunner(QueryReformulationRunner):
    config_type = QueryReformulationBenchmarkConfig

    @staticmethod
    def create_model(config, path_prefix):
        print(BackwardCompatibilityQueryReformulationRunner.old_model_path)
        model = bolt.UniversalDeepTransformer.load(
            BackwardCompatibilityQueryReformulationRunner.old_model_path
        )

        return model

    @classmethod
    def run_benchmark(
        cls, config: QueryReformulationBenchmarkConfig, path_prefix: str, mlflow_logger
    ):
        config.dataset_size = "small"

        filtered_versions = get_filtered_versions()

        failed_versions = []
        for filtered_version in filtered_versions:
            print(filtered_version)
            formatted_version = filtered_version.replace(".", "_")
            BackwardCompatibilityQueryReformulationRunner.old_model_path = os.path.join(
                OLD_MODEL_PATH, f"{formatted_version}_{config.config_name}.model"
            )
            try:
                QueryReformulationRunner.run_benchmark.__func__(
                    BackwardCompatibilityQueryReformulationRunner,
                    config,
                    path_prefix,
                    mlflow_logger,
                )
            except Exception as error:
                failed_versions.append(filtered_version)
                print(
                    f"An error occurred running the {config.config_name} benchmark with version {filtered_version}:",
                    error,
                )

        if failed_versions:
            raise Exception(
                f"The {config.config_name} benchmark failed when loading the following versions: {', '.join(failed_versions)}"
            )
