from ...configs.ndb_configs import *
from ..ndb_runner import NDBRunner


class MiniBenchmarkNDBRunner(NDBRunner):
    config_type = NDBConfig

    @classmethod
    def run_benchmark(cls, config: NDBConfig, path_prefix: str, mlflow_logger):
        NDBRunner.run_benchmark(config, path_prefix, mlflow_logger)
