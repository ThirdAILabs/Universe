import os

import ray
import thirdai.distributed_bolt as db
from ray.cluster_utils import Cluster
from thirdai import bolt

from ..configs.distributed_configs import DistributedBenchmarkConfig
from .runner import Runner


def create_udt_model(n_target_classes, output_dim, num_hashes):
    model = bolt.UniversalDeepTransformer(
        data_types={
            "QUERY": bolt.types.text(contextual_encoding="local"),
            "DOC_ID": bolt.types.categorical(delimiter=":"),
        },
        target="DOC_ID",
        n_target_classes=n_target_classes,
        integer_target=True,
        options={
            "extreme_classification": True,
            "train_without_bias": True,
            "embedding_dimension": 2048,
            "freeze_hash_tables": False,
            "extreme_output_dim": output_dim,
            "extreme_num_hashes": num_hashes,
        },
    )
    model._get_model().summary()

    return model


def make_cluster_config(num_cpu_per_node, cluster_address, communication_type="linear"):
    # We set the working_dir for the cluster equal to this directory
    # so that pickle works. Otherwise, unpickling functions
    # defined in the test files would not work, since pickle needs to be
    # able to import the file the object/function was originally defined in.

    working_dir = os.path.dirname(os.path.realpath(__file__))
    cluster_config = db.RayTrainingClusterConfig(
        num_workers=2,
        requested_cpus_per_node=num_cpu_per_node,
        communication_type=communication_type,
        cluster_address=cluster_address,
        runtime_env={"working_dir": working_dir},
        ignore_reinit_error=True,
    )
    return cluster_config


def initilize_ray_two_node_cluster():
    # ============ Initilize a two_node_cluster ===============
    num_cpu_per_node = db.get_num_cpus() // 2

    # case if multiprocessing import fails
    if num_cpu_per_node == 0:
        num_cpu_per_node = 1

    mini_cluster = Cluster(
        initialize_head=True,
        head_node_args={
            "num_cpus": num_cpu_per_node,
        },
    )
    mini_cluster.add_node(num_cpus=num_cpu_per_node)
    return num_cpu_per_node, mini_cluster
    # ================ Initialized a cluster ==================


def destroy_cluster(mini_cluster):
    # =============== Destroying cluster ======================
    ray.shutdown()
    mini_cluster.shutdown()
    # =============== Destroyed cluster =======================


class DistributedRunner(Runner):
    config_type = DistributedBenchmarkConfig

    def run_benchmark(config: DistributedBenchmarkConfig, path_prefix, mlflow_logger):
        # prepare dataset
        config.prepare_dataset(path_prefix=path_prefix)

        # Initilize ray cluster
        num_cpu_per_node, mini_cluster = initilize_ray_two_node_cluster()

        # Create model
        model = create_udt_model(
            n_target_classes=config.n_target_classes,
            output_dim=config.output_dim,
            num_hashes=config.num_hashes,
        )
        # print("======== Model Created ===========")
        validation = bolt.Validation(
            os.path.join(path_prefix, config.supervised_tst),
            interval=5000,
            metrics=config.val_metrics,
        )

        # curr_dir = os.getcwd()
        # print(f"{os.getcwd()} is the current working directory.")

        # print("========= Training started ==========")
        if hasattr(config, "supervised_trn_1"):
            model.train_distributed(
                cluster_config=make_cluster_config(
                    num_cpu_per_node=num_cpu_per_node,
                    cluster_address=mini_cluster.address,
                    communication_type="linear",
                ),
                filenames=[
                    os.path.join(path_prefix, config.supervised_trn_1),
                    os.path.join(path_prefix, config.supervised_trn_2),
                ],
                learning_rate=config.learning_rate,
                epochs=config.num_epochs,
                metrics=config.train_metrics,
                validation=validation,
                # callbacks=[LoggingCallback(model, supervised_tst)],
            )

        # print("========== Cold Start Training =============")
        if hasattr(config, "unsupervised_file_1"):
            model.cold_start_distributed(
                cluster_config=make_cluster_config(
                    num_cpu_per_node=num_cpu_per_node,
                    cluster_address=mini_cluster.address,
                    communication_type="linear",
                ),
                filenames=[
                    os.path.join(path_prefix, config.unsupervised_file_1),
                    os.path.join(path_prefix, config.unsupervised_file_2),
                ],
                strong_column_names=["TITLE"],
                weak_column_names=["TEXT"],
                learning_rate=config.learning_rate,
                epochs=20,
                metrics=[
                    "precision@1",
                    "recall@10",
                ],
                # callbacks=[LoggingCallback(model, supervised_tst)],
            )

        # Destroy the cluster
        destroy_cluster(mini_cluster)
