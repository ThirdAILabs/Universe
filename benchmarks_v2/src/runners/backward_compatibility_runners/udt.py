from ...configs.cold_start_configs import *
from ...configs.graph_configs import *
from ...configs.mach_configs import *
from ...configs.udt_configs import *
from ..udt import UDTRunner
from thirdai import bolt, deployment
import json
import requests
from distutils.version import StrictVersion

def versions(package_name):
    url = "https://pypi.org/pypi/%s/json" % (package_name,)
    data = json.loads(requests.get(url).content)
    versions = list(data["releases"].keys())
    versions.sort(key=StrictVersion, reverse=True)
    return versions


class BackwardCompatibilityUDTRunner(UDTRunner):
    config_type = UDTBenchmarkConfig
    old_model_path = ""

    @staticmethod
    def create_model(config, path_prefix):
        print('HEREEEEEEEEEEEEEEEEEEEEE')
        print(BackwardCompatibilityUDTRunner.old_model_path)
        model = bolt.UniversalDeepTransformer.load(BackwardCompatibilityUDTRunner.old_model_path)
        print('HEREEEEEEEEEEEEEEEEEEEEE')

        return model

    @classmethod
    def run_benchmark(cls, config: UDTBenchmarkConfig, path_prefix: str, mlflow_logger):
        if config.num_epochs:
            config.num_epochs = 1
        if config.cold_start_num_epochs:
            config.cold_start_num_epochs = 1

        test_versions = versions("thirdai")[1:2]

        BackwardCompatibilityUDTRunner.old_model_path = "test_udt.model"

        UDTRunner.run_benchmark.__func__(BackwardCompatibilityUDTRunner, config, path_prefix, mlflow_logger)

        # getattr(BackwardCompatibilityUDTRunner, UDTRunner.run_benchmark)(config, path_prefix, mlflow_logger)
        
