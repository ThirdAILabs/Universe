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


class MiniBenchmarkUDTRunner(UDTRunner):
    config_type = UDTBenchmarkConfig

    @staticmethod
    def create_model(config, path_prefix):
        if config.model_config is not None:
            model_config_path = config.config_name + "_model.config"
            deployment.dump_config(
                config=json.dumps(config.model_config),
                filename=model_config_path,
            )
        else:
            model_config_path = None

        data_types = config.get_data_types(path_prefix)
        model = bolt.UniversalDeepTransformer(
            data_types=data_types,
            target=config.target,
            integer_target=config.integer_target,
            n_target_classes=config.n_target_classes,
            temporal_tracking_relationships=config.temporal_relationships,
            delimiter=config.delimiter,
            model_config=model_config_path,
            options=config.options,
        )

        if model_config_path:
            os.remove(model_config_path)

        return model

    @classmethod
    def run_benchmark(cls, config: UDTBenchmarkConfig, path_prefix: str, mlflow_logger):
        if config.num_epochs:
            config.num_epochs = 1
        if config.cold_start_num_epochs:
            config.cold_start_num_epochs = 1

        test_versions = versions("thirdai")[1:2]
        
