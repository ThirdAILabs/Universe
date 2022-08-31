from thirdai._distributed_bolt._private.distributed_bolt import DistributedBolt
import ray
import os
import toml
import textwrap
from thirdai._distributed_bolt._private.primary_worker import PrimaryWorker
from thirdai._distributed_bolt._private.replica_worker import ReplicaWorker
from thirdai._distributed_bolt._models.model_type import ModelType
from .utils import get_num_cpus, init_logging
from typing import Tuple, Any, Optional, Dict, List


class FullyConnectedNetwork(DistributedBolt):
    """This class implements the public facing APIs for
    Fully Connected Network Class.

    Args:
        DistributedBolt (Class): Implements the generic class for
        Public Facing APIs which includes functions like train, predict
    """

    def __init__(self, no_of_workers, config_filename, num_cpus_per_node = -1, num_omp_threads_experiment=-1):
        """This function initializes this class, which provides wrapper over DistributedBolt and
        implements the user facing FullyConnectedNetwork API.

        Args:
            no_of_workers (int): number of workers
            config_filename (dict): configuration file for FullyConnectedNetwork
            num_cpus_per_node (int): Number of CPUs per node

        Raises:
            ValueError: If number of training files is not equal to number of nodes
            Exception: If ray initialization doesnot happens
        """

        self.logging = init_logging("distributed_fully_connected.log")
        self.logging.info("Training has started!")

        try:
            config = toml.load(config_filename)
        except Exception:
            self.logging.error(
                "Could not load the toml file! "
                + "Config File Location:"
                + config_filename
            )

        if len(config["dataset"]["train_data"]) != no_of_workers:
            raise ValueError(
                "Received ",
                str(len(config["dataset"]["train_data"])),
                " training datasets. Expected ",
                no_of_workers,
                " datasets, one for each node.",
            )

        self.no_of_workers = no_of_workers

        # setting OMP_NUM_THREADS to number of num_cpus
        if num_omp_threads_experiment is not -1:
            num_omp_threads = str(num_omp_threads_experiment)
        else:
            num_omp_threads = get_num_cpus()
        self.logging.info("Setting OMP_NUM_THREADS to " + num_omp_threads)
        runtime_env = {"env_vars": {"OMP_NUM_THREADS": str(num_omp_threads)}}

        ray.init(address="auto", runtime_env=runtime_env)
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

        self.model_type = ModelType.FullyConnectedNetwork
        self.epochs = config["params"]["epochs"]
        self.learning_rate = config["params"]["learning_rate"]
        self.num_layers = len(config["layers"])+1

        num_cpus = get_num_cpus()
        if num_cpus_per_node is not -1:
            num_cpus = num_cpus_per_node

        

        self.primary_worker = PrimaryWorker.options(
            num_cpus=num_cpus, max_concurrency=100
        ).remote(
            self.num_layers, config, self.no_of_workers, self.model_type
        )

        self.replica_workers = [
            ReplicaWorker.options(num_cpus=num_cpus, max_concurrency=100).remote(
                self.num_layers,
                config,
                self.no_of_workers,
                worker_id + 1,
                self.primary_worker,
                self.model_type,
            )
            for worker_id in range(self.no_of_workers - 1)
        ]

        self.workers = [self.primary_worker]
        self.workers.extend(self.replica_workers)

        ray.get([worker.make_fully_connected_model.remote() for worker in self.workers])

        self.primary_worker.add_workers.remote(self.workers)

        self.num_of_batches = min(
            ray.get([worker.num_of_batches.remote() for worker in self.workers])
        )

        # updating weights and parameters across all the nodes
        ray.get([worker.synchronize_parameters.remote() for worker in self.workers])
        super().__init__(
            self.workers,
            self.logging,
            self.epochs,
            self.primary_worker,
            self.num_of_batches,
            self.model_type,
        )
