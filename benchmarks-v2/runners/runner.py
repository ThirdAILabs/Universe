from abc import ABC, abstractmethod


class Runner(ABC):
    @property
    @staticmethod
    @abstractmethod
    def config_type():
        """
        Returns the type of config supported by the given runner.
        """
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def run_benchmark(config, path, mlflow_logger=None):
        raise NotImplementedError
