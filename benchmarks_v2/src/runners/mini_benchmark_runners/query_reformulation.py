from ...configs.query_reformulation_configs import *
from ..query_reformulation import QueryReformulationRunner


class MiniBenchmarkQueryReformulationRunner(QueryReformulationRunner):
    config_type = QueryReformulationBenchmarkConfig

    @classmethod
    def run_benchmark(
        cls, config: QueryReformulationBenchmarkConfig, path_prefix: str, mlflow_logger
    ):
        config.dataset_size = "small"

        QueryReformulationRunner.run_benchmark(config, path_prefix, mlflow_logger)
