from thirdai._distributed_bolt._private.distributed_bolt import DistributedBolt
import ray
import os
import toml
import textwrap
from thirdai._distributed_bolt._private.primary_worker import PrimaryWorker
from thirdai._distributed_bolt._private.replica_worker import ReplicaWorker
from .utils import get_num_cpus, init_logging
from typing import Tuple, Any, Optional, Dict, List


class FullyConnectedNetwork(DistributedBolt):
    def __init__(
        self,
        no_of_workers,
        config_filename,
        num_cpus_per_node
    ):

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
        num_omp_threads = str(get_num_cpus())

        self.logging.info("Setting OMP_NUM_THREADS to " + num_omp_threads)
        runtime_env = {"env_vars": {"OMP_NUM_THREADS": str(get_num_cpus())}}

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

        self.epochs = config["params"]["epochs"]
        self.learning_rate = config["params"]["learning_rate"]
        self.layer_dims = [config["dataset"]["input_dim"]]

        for i in range(len(config["layers"])):
            self.layer_dims.append(config["layers"][i]["dim"])

        num_cpus = get_num_cpus()
        if num_cpus_per_node is not -1:
            num_cpus = num_cpus_per_node

        self.primary_worker = PrimaryWorker.options(
            num_cpus=num_cpus, max_concurrency=100
        ).remote(self.layer_dims, config, self.no_of_workers)

        self.replica_workers = [
            ReplicaWorker.options(num_cpus=num_cpus, max_concurrency=100).remote(
                self.layer_dims,
                config,
                self.no_of_workers,
                worker_id + 1,
                self.primary_worker,
            )
            for worker_id in range(self.no_of_workers - 1)
        ]

        self.workers = [self.primary_worker]
        self.workers.extend(self.replica_workers)

        self.primary_worker.add_workers.remote(self.workers)

        self.num_of_batches = min(
            ray.get([worker.num_of_batches.remote() for worker in self.workers])
        )

        # updating weights and parameters across all the nodes
        ray.get([worker.synchronize_parameters.remote() for worker in self.workers])
        super().__init__(self.workers, self.logging, self.epochs, self.primary_worker, self.num_of_batches)
