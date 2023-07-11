import os

from thirdai import bolt

from ..configs.distributed_configs import DistributedBenchmarkConfig
from ..distributed_utils import ray_two_node_cluster_config
from .runner import Runner


def create_udt_model(n_target_classes, output_dim, num_hashes, embedding_dimension):
    model = bolt.UniversalDeepTransformer(
        data_types={
            "QUERY": bolt.types.text(contextual_encoding="local"),
            "DOC_ID": bolt.types.categorical(delimiter=":"),
        },
        target="DOC_ID",
        n_target_classes=n_target_classes,
        integer_target=True,
        options={
            "embedding_dimension": embedding_dimension,
            "extreme_output_dim": output_dim,
            "extreme_num_hashes": num_hashes,
            "use_bias": True,
        },
    )
    return model


class DistributedRunner(Runner):
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

        if hasattr(config, "unsupervised_file_1"):
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

        if hasattr(config, "supervised_trn_1"):
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
