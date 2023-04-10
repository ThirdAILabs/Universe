import json
import os

from thirdai import bolt, deployment

from ..configs.query_reformulation_configs import QueryReformulationBenchmarkConfig
from .runner import Runner


class QueryReformulationRunner(Runner):
    config_type = QueryReformulationBenchmarkConfig

    def run_benchmark(config: QueryReformulationBenchmarkConfig, path_prefix: str, mlflow_logger):
        train_file = os.path.join(path_prefix, config.train_file)
        test_file = os.path.join(path_prefix, config.test_file)

        if config.model_config is not None:
            model_config_path = config.config_name + "_model.config"
            deployment.dump_config(
                config=json.dumps(config.model_config),
                filename=model_config_path,
            )
        else:
            model_config_path = None

        model = bolt.UniversalDeepTransformer(
            source_column=config.source_column, target_column=config.target_column, dataset_size=config.dataset_size
        )

        if model_config_path:
            os.remove(model_config_path)

        model.train(train_file)

        for metric_name, metric_fn in config.additional_metric_fns.items():
            metric_val = metric_fn(model, test_file)

            print(f"{metric_name} = {metric_val}")
            mlflow_logger.log_additional_metric(key=metric_name, value=metric_val, step=0)
