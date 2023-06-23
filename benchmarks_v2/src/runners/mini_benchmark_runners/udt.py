from ...configs.cold_start_configs import *
from ...configs.graph_configs import *
from ...configs.mach_configs import *
from ...configs.udt_configs import *
from ..udt import UDTRunner


class MiniBenchmarkUDTRunner(UDTRunner):
    config_type = UDTBenchmarkConfig

    @classmethod
    def run_benchmark(cls, config: UDTBenchmarkConfig, path_prefix: str, mlflow_logger):
        if config.num_epochs:
            config.num_epochs = 1
        if config.cold_start_num_epochs:
            config.cold_start_num_epochs = 1

        # The ROC_AUC additional metric lead to errors in the mini benchmarks because
        # all of the predictions are the same which leads to ROC_AUC being undefined.
        if config.callbacks:
            config.callbacks = []

        UDTRunner.run_benchmark(config, path_prefix, mlflow_logger)
