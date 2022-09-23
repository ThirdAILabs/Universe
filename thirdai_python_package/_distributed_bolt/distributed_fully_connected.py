import string
from thirdai._distributed_bolt.backend.distributed_bolt import DistributedBolt
import ray
import os
import toml
import textwrap
from thirdai._distributed_bolt.backend.primary_worker import PrimaryWorker
from thirdai._distributed_bolt.backend.replica_worker import ReplicaWorker
from .utils import get_num_cpus, init_logging
from typing import Tuple, Any, Optional, Dict, List


class FullyConnectedNetwork(DistributedBolt):
    """
    This class implements the public facing APIs for
    Fully Connected Network Class.

    :param DistributedBolt: Implements the generic class for
                Public Facing APIs which includes functions like train, predict
    :type DistributedBolt: DistributedBolt
    """

    def __init__(
        self,
        num_workers,
        config_filename,
        num_cpus_per_node: Optional[int] = -1,
        communication_type: Optional[str] = "circular",
        cluster_address: Optional[str] = "auto",
    ):
        """
        This function initializes this class, which provides wrapper over DistributedBolt and
        implements the user facing FullyConnectedNetwork API.

        :param num_workers: number of workers
        :type num_workers: int
        :param config_filename: configuration file for FullyConnectedNetwork
        :type config_filename: Dict
        :param num_cpus_per_node: Number of CPUs per node, defaults to -1
        :type num_cpus_per_node: Optional[int], optional
        :param communication_type: type of communication, defaults to "circular"
        :type communication_type: Optional[str], optional
        :param cluster_address: address to which cluster needed to add, defaults to "auto"
        :type cluster_address: Optional[str], optional
        :raises ValueError: If number of training files is not equal to number of nodes
        :raises Exception: If ray initialization doesnot happens
        """

        self.logging = init_logging("distributed_fully_connected.log")
        self.logging.info("Training has started!")

        # We do not need a try catch statements here,
        # toml error prompts are enough
        config = toml.load(config_filename)

        if len(config["dataset"]["train_data"]) != num_workers:
            raise ValueError(
                "Received ",
                str(len(config["dataset"]["train_data"])),
                " training datasets. Expected ",
                num_workers,
                " datasets, one for each node.",
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

        self.logging.info("Ray Initialized")

        self.epochs = config["params"]["epochs"]
        self.learning_rate = config["params"]["learning_rate"]
        self.layer_dims = []

        for i in range(len(config["nodes"])):
            self.layer_dims.append(config["nodes"][i]["dim"])

        # num_cpus_per_worker is checking num_cpus_per_node function
        # parameter, that whether user wants  to use some particular number of
        # CPUs per worker to be used else, they would be detected automatically

        num_cpus_per_worker = get_num_cpus()
        if num_cpus_per_node != -1:
            if num_cpus_per_node <= num_cpus_per_worker:
                num_cpus_per_worker = num_cpus_per_node
            else:
                raise ValueError(
                    "Argument num_cpus_per_node=",
                    num_cpus_per_node,
                    "could not be greater than number of cpus on machine, which is",
                    num_cpus_per_worker,
                )

        # max_concurrency here, indicates the number of threads
        # that this particular worker can run. Setting it a large value like
        # 100, as the ray queues the work load else.
        self.primary_worker = PrimaryWorker.options(
            num_cpus=num_cpus_per_worker, max_concurrency=100
        ).remote(self.layer_dims, self.num_workers, config, communication_type)

        self.replica_workers = [
            ReplicaWorker.options(
                num_cpus=num_cpus_per_worker, max_concurrency=100
            ).remote(
                self.num_workers,
                worker_id + 1,
                self.primary_worker,
                config,
                self.layer_dims,
                communication_type,
            )
            for worker_id in range(self.num_workers - 1)
        ]

        self.workers = [self.primary_worker]
        self.workers.extend(self.replica_workers)

        self.num_of_batches = min(
            ray.get([worker.num_of_batches.remote() for worker in self.workers])
        )

        ray.get([worker.synchronize_parameters.remote() for worker in self.workers])
        super().__init__(
            self.workers,
            self.logging,
            self.epochs,
            self.primary_worker,
            self.num_of_batches,
            communication_type,
        )
