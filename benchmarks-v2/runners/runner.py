import json
import os
from abc import ABC, abstractmethod

from thirdai import bolt, deployment

from ..utils import check_if_experimental_mode


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
    def run_benchmark(config, path_prefix, mlflow_logger=None):
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def get_average_predict_time(
        model, test_file, config, path_prefix, num_samples=10000
    ):
        raise NotImplementedError

    @staticmethod
    def create_model(config, path_prefix):

        experimental_autotune = check_if_experimental_mode()

        if experimental_autotune:
            print("In Experimental Autotune Mode")
        else:
            print("In Autotuned Sparsity Mode")

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

        boltmodel = model._get_model()
        from .utils import add_sparsity_to_first_non_sparse_layer

        print("Model summary")
        boltmodel.summary()
        layer = add_sparsity_to_first_non_sparse_layer(
            boltmodel, experimental_autotune=experimental_autotune
        )
        print(f"Model summary after changing the sparsity for the layer fc_{layer}")
        boltmodel.summary()

        if model_config_path:
            os.remove(model_config_path)

        return model
