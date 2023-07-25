import os

from thirdai import bolt

from ..configs.distributed_configs import DistributedBenchmarkConfig
from ..distributed_utils import create_udt_model, ray_two_node_cluster_config
from .runner import Runner


class DistributedRunner_v1(Runner):
    config_type = DistributedBenchmarkConfig

    def run_benchmark(config: DistributedBenchmarkConfig, path_prefix, mlflow_logger):
        # prepare dataset
        config.prepare_dataset(path_prefix=path_prefix)

        # Initilize ray cluster
        cluster_generator_obj = ray_two_node_cluster_config()
        cluster_config_fn = next(cluster_generator_obj)

        # Create model
        model = create_udt_model(
            n_target_classes=config.n_target_classes,
            output_dim=config.output_dim,
            num_hashes=config.num_hashes,
            embedding_dimension=config.embedding_dimension,
        )

        validation = bolt.Validation(
            filename=os.path.join(path_prefix, config.supervised_tst),
            interval=2,
            metrics=config.val_metrics,
        )

        metrics = model.cold_start_distributed(
            cluster_config=cluster_config_fn(communication_type="linear"),
            filenames=[
                os.path.join(path_prefix, config.unsupervised_file_1),
                os.path.join(path_prefix, config.unsupervised_file_2),
            ],
            batch_size=8192,
            strong_column_names=["TITLE"],
            weak_column_names=["TEXT"],
            learning_rate=config.learning_rate,
            epochs=config.num_epochs,
            metrics=config.train_metrics,
        )

        if config.supervised_trn_1:
            metrics = model.train_distributed(
                cluster_config=cluster_config_fn(communication_type="linear"),
                filenames=[
                    os.path.join(path_prefix, config.supervised_trn_1),
                    os.path.join(path_prefix, config.supervised_trn_2),
                ],
                batch_size=8192,
                learning_rate=config.learning_rate,
                epochs=config.num_epochs,
                metrics=config.train_metrics,
                validation=validation,
            )

        # Destroy the cluster
        next(cluster_generator_obj)
