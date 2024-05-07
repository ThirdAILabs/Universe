from ...configs.temporal_configs import *
from ..temporal import TemporalRunner


class MiniBenchmarkTemporalRunner(TemporalRunner):
    config_type = TemporalBenchmarkConfig

    @classmethod
    def run_benchmark(
        cls, config: TemporalBenchmarkConfig, path_prefix: str, mlflow_logger
    ):
        if config.num_epochs:
            config.num_epochs = 1

        TemporalRunner.run_benchmark(config, path_prefix, mlflow_logger)
