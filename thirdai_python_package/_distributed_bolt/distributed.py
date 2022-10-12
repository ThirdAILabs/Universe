import os
import tempfile
import textwrap
from typing import List

import ray
from thirdai._distributed_bolt.backend.communication import AVAILABLE_METHODS
from thirdai._distributed_bolt.backend.primary_worker import PrimaryWorker
from thirdai._distributed_bolt.backend.replica_worker import ReplicaWorker
from thirdai._distributed_bolt.backend.train_state_manager import TrainStateManager
from thirdai._thirdai import bolt, logging

from .utils import get_num_cpus, init_logging


class RayTrainingClusterConfig:
    """
    The RayTrainingClusterConfig object represents an initialized Ray cluster
    that we know will work for training (worker and head nodes initialized,
    logging initialized, etc.).
    """

    def __init__(
        self,
        num_workers: int,
        requested_cpus_per_node: int = -1,
        communication_type: str = "circular",
        cluster_address: str = "auto",
        log_dir: str = os.path.join(tempfile.gettempdir(), "thirdai"),
    ):
        """
        This constructor connects to an already existing Ray cluster,
        starts Ray workers on each node, initializes logging, and creates
        Ray primary and replica worker configs. It computes and stores a
        a number of useful fields, including num_workers, communication_type,
        logging, primary_worker_config, and replica_worker_configs.
        """
        if not os.path.exists(log_dir):
            os.mkdir(log_dir)

        distributed_training_log_file = os.path.join(log_dir, "distributed_bolt.log")

        self.logging = init_logging(distributed_training_log_file)
        self.log_dir = log_dir
        self.logging.info("Building Ray training cluster")
        self.communication_type = communication_type

        if self.communication_type not in AVAILABLE_METHODS:
            raise ValueError(
                textwrap.dedent(
                    """
                        Currently only three modes of communication are supported.
                        Use: "circular" or "linear" or "gloo". 
                    """
                )
            )

        self.num_workers = num_workers

        # setting OMP_NUM_THREADS to number of num_cpus
        # Ray expicitly forces the OMP_NUM_THREADS in environment to 1.
        # So, we need to change the OMP_NUM_THREADS to support parallization
        num_omp_threads = str(get_num_cpus())
        self.logging.info("Setting OMP_NUM_THREADS to " + num_omp_threads)
        runtime_env = {"env_vars": {"OMP_NUM_THREADS": str(get_num_cpus())}}

        ray.init(address=cluster_address, runtime_env=runtime_env)
        if not ray.is_initialized():
            raise Exception(
                textwrap.dedent(
                    """
                Some issue with cluster setup. Ray is not getting initialized.
                Make sure to have ray cluster online before calling
                Distributed Bolt.
            """
                )
            )

        self.logging.info("Connected to Ray cluster!")

        num_cpus_on_this_node = get_num_cpus()
        if requested_cpus_per_node != -1:
            num_cpus_to_use = min(requested_cpus_per_node, num_cpus_on_this_node)
        else:
            num_cpus_to_use = num_cpus_on_this_node

        self.logging.info(
            f"Using {num_cpus_to_use} cpus / node (user requested {requested_cpus_per_node})"
        )

        # TODO(Josh/Pratik): investigate the correct setting for max concurrency
        self.primary_worker_config = PrimaryWorker.options(
            num_cpus=num_cpus_to_use, max_concurrency=100
        )

        self.replica_worker_configs = [
            ReplicaWorker.options(num_cpus=num_cpus_to_use, max_concurrency=100)
            for _ in range(self.num_workers - 1)
        ]


class DistributedDataParallel:
    """
    This class implements the public facing APIs for a distributed data parallel
    model.
    """

    def __init__(
        self,
        cluster_config: RayTrainingClusterConfig,
        model: bolt.graph.Model,
        train_config: bolt.graph.TrainConfig,
        train_file_names: List[str],
        batch_size: int,
    ):
        """
        This constructor returns a new DistributedDataParallel object that can
        be used to train the given model in a distributed fashion on the cluster
        corresponding to the passed in cluster_config. This constructor also
        passes the given model, the training config, and the corresponding
        training file name to each node in the cluster, thereby ensuring that
        each node is ready for training. After this constructor returns, the
        user can simply call train to train the model on the cluster.
        """
        self.communication_type = cluster_config.communication_type
        self.logging = cluster_config.logging
        self.train_config = train_config

        if len(train_file_names) != cluster_config.num_workers:
            raise ValueError(
                "Received ",
                len(train_file_names),
                " training datasets. Expected ",
                cluster_config.num_workers,
                " datasets, one for each node.",
            )

        self.logging.info("Training has started!")

        # This speeds up passing the complete model to each worker by having
        # Ray serialize it once and save it in the object store instead of
        # serializing it for every worker individually. See
        # https://docs.ray.io/en/latest/ray-core/tips-for-first-time.html#tip-3-avoid-passing-same-object-repeatedly-to-remote-tasks
        # for more details.
        ray_model_ref = ray.put(model)

        self.primary_worker = cluster_config.primary_worker_config.remote(
            num_workers=cluster_config.num_workers,
            model_to_wrap=ray_model_ref,
            train_file_name=train_file_names[0],
            train_config=train_config,
            communication_type=cluster_config.communication_type,
            log_dir=cluster_config.log_dir,
            batch_size=batch_size,
        )

        self.replica_workers = []
        for worker_id, replica_worker_config in enumerate(
            cluster_config.replica_worker_configs
        ):
            self.replica_workers.append(
                replica_worker_config.remote(
                    num_workers=cluster_config.num_workers,
                    model_to_wrap=ray_model_ref,
                    train_file_name=train_file_names[worker_id + 1],
                    train_config=train_config,
                    id=worker_id + 1,
                    primary_worker=self.primary_worker,
                    communication_type=cluster_config.communication_type,
                    log_dir=cluster_config.log_dir,
                    batch_size=batch_size,
                )
            )

        self.workers = [self.primary_worker] + self.replica_workers

        self.num_of_batches = min(
            ray.get([worker.num_of_batches.remote() for worker in self.workers])
        )

        self.logging.info(
            f"Data loaded on all nodes, minimmum num batches is {self.num_of_batches}."
        )

    def train(self) -> None:
        """
        Trains the network using the communication type choosen.
        """
        train_state_manager = TrainStateManager(
            self.workers,
            self.primary_worker,
            self.logging,
            self.communication_type,
        )

        for epoch in range(self.train_config.num_epochs):
            for batch_id in range(self.num_of_batches):

                # Here we are asking every worker to calculate their gradients and return
                # once they all calculate their gradients
                train_state_manager.train_batch(epoch, batch_id)

        train_state_manager.finish_training()

    def get_model(self, worker_id=0):
        return ray.get(self.workers[worker_id].model.remote())
