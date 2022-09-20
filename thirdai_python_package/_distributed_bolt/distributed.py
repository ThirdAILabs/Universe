import string
from thirdai._thirdai import bolt, dataset
from thirdai._distributed_bolt.backend.distributed_bolt import DistributedBolt
import ray
import textwrap
from thirdai._distributed_bolt.backend.primary_worker import PrimaryWorker
from thirdai._distributed_bolt.backend.replica_worker import ReplicaWorker
from .utils import get_num_cpus, init_logging
from typing import Tuple, Any, Optional, Dict, List


class RayTrainingCluster:
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

        self.logging.info("Ray Initialized")

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
        cluster: RayTrainingCluster,
        model: bolt.graph.Model,
        train_config: bolt.graph.TrainConfig,
        train_file_names: List[str],
    ):

        self.cluster = cluster
        self.logging = cluster.logging

        if len(train_file_names) != cluster.num_workers:
            raise ValueError(
                "Received ",
                len(train_file_names),
                " training datasets. Expected ",
                cluster.num_workers,
                " datasets, one for each node.",
            )

        self.logging.info("Training has started!")

        self.primary_worker = cluster.primary_worker_config.remote(
            cluster.num_workers,
            model,
            train_file_names[0],
            # train_config,
            cluster.communication_type,
        )

        self.replica_workers = []
        for worker_id, replica_worker_config in enumerate(cluster.replica_worker_configs):
            self.replica_workers.append(replica_worker_config.remote(
                cluster.num_workers,
                worker_id + 1,
                self.primary_worker,
                model,
                train_file_names[worker_id + 1],
                # train_config,
                cluster.communication_type,
            ))

        self.num_of_batches = min(
            ray.get([worker.num_of_batches.remote() for worker in self.replica_workers])
        )

    def train(self):
        pass

    def get_model(self):
        return self.cluster.primary_worker.model.remote()
