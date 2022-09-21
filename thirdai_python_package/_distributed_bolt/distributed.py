from thirdai._thirdai import bolt
from thirdai._distributed_bolt.backend.communication import AVAILABLE_METHODS
from thirdai._distributed_bolt.backend.trainer import Trainer
import ray
import textwrap
from thirdai._distributed_bolt.backend.primary_worker import PrimaryWorker
from thirdai._distributed_bolt.backend.replica_worker import ReplicaWorker
from .utils import get_num_cpus, init_logging
from typing import Optional, List


class RayTrainingClusterConfig:
    def __init__(
        self,
        num_workers: int,
        requested_cpus_per_node: Optional[int] = -1,
        communication_type: Optional[str] = "circular",
        cluster_address: Optional[str] = "auto",
    ):

        self.logging = init_logging("distributed_fully_connected.log")
        self.logging.info("Building Ray training cluster")
        self.communication_type = communication_type

        if self.communication_type not in AVAILABLE_METHODS:
            raise ValueError(
                textwrap.dedent(
                    """
                        Currently only two modes of communication is supported.
                        Use: "circular" or "linear". 
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

        # TODO(Josh): investigate this max concurrency thing
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
    ):

        self.communication_type = cluster_config.communication_type
        self.logging = cluster_config.logging

        if len(train_file_names) != cluster_config.num_workers:
            raise ValueError(
                "Received ",
                len(train_file_names),
                " training datasets. Expected ",
                cluster_config.num_workers,
                " datasets, one for each node.",
            )

        self.logging.info("Training has started!")

        self.primary_worker = cluster_config.primary_worker_config.remote(
            num_workers=cluster_config.num_workers,
            model_to_wrap=model,
            train_file_name=train_file_names[0],
            train_config=train_config,
            communication_type=cluster_config.communication_type,
        )

        self.replica_workers = []
        for worker_id, replica_worker_config in enumerate(
            cluster_config.replica_worker_configs
        ):
            self.replica_workers.append(
                replica_worker_config.remote(
                    num_workers=cluster_config.num_workers,
                    model_to_wrap=model,
                    train_file_name=train_file_names[worker_id + 1],
                    train_config=train_config,
                    id=worker_id + 1,
                    primary_worker=self.primary_worker,
                    communication_type=cluster_config.communication_type,
                )
            )

        self.workers = [self.primary_worker] + self.replica_workers

        self.num_of_batches = min(
            ray.get([worker.num_of_batches.remote() for worker in self.workers])
        )

        print("Num batches,", self.num_of_batches)

        self.logging.info(
            f"Data loaded on all nodes, minimmum num batches is {self.num_of_batches}."
        )

    def train(self) -> None:
        """
        Trains the network using the communication type choosen.
        """
        trainer = Trainer(
            self.workers,
            self.primary_worker,
            self.logging,
            self.communication_type,
        )

        # TODO(Josh): Fix epochs
        for epoch in range(1):
            for batch_id in range(self.num_of_batches):

                # Here we are asking every worker to calculate their gradients and return
                # once they all calculate their gradients
                trainer.train(epoch, batch_id)

        trainer.finish_training()

    def get_model(self):
        return ray.get(self.primary_worker.model.remote())
