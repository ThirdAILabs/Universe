import ray
import os
import toml
import textwrap
from .primary_worker import PrimaryWorker
from .replica_worker import ReplicaWorker
import time as time
from .utils import get_num_cpus, init_logging
from typing import Tuple, Any, Optional, Dict, List


class DistributedBolt:
    """Implements all the user level Distributed Bolt APIs to the users."""

    def __init__(
        self,
        no_of_workers: int,
        config_filename: str,
        num_cpus_per_node: Optional[int] = -1,
    ):
        """Initializes the DistributeBolt class.

        Args:
            no_of_workers (int): Number of workers to start training on.
            This number should be less than equal to the number of nodes(including the head node) training
            is started.
            config_filename (str): The name of the config file which is going to be used for training.
            num_cpus_per_node (Optional[int], optional): Number of CPUs to be used on a node. Default Value = number of cpus on current
            node. Defaults to -1.

        Raises:
            ValueError: Number of Dataset not equal to number of nodes
            Exception: Ray Cluster not started.
        """

        self.logging = init_logging("DistributedBolt.log")
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

        # check for whether OMP_NUM_THREADS already set by user
        num_omp_threads = str(get_num_cpus())
        # if "OMP_NUM_THREADS" in os.environ:
        #     num_omp_threads = os.environ["OMP_NUM_THREADS"]
        #     self.logging.warning(
        #         "Reading OMP_NUM_THREADS from environment to be " + num_omp_threads
        #     )
        #     self.logging.warning(
        #         "To use default OMP_NUM_THREADS, try running the program in new shell, or update the OMP_NUM_THREADS in the current environment"
        #     )

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

        self.bolt_computation_time = 0
        self.averaging_and_communication_time = 0

    def train(self, circular: Optional[bool] = True) -> None:
        """Trains the network using the communication type choosen.

        Args:
            circular (Optional[bool], optional): True, if circular communication is required.
                    False, if linear communication is required.. Defaults to True.
        """

        # initial configuration for circular communcation
        if circular:
            self.logging.info("Circular communication pattern is choosen")
            for i in range(len(self.workers)):
                ray.get(
                    self.workers[i].add_friend.remote(
                        self.workers[(i - 1) % (len(self.workers))]
                    )
                )
        else:
            self.logging.info("Linear communication pattern is choosen")

        for epoch in range(self.epochs):

            for batch_no in range(int(self.num_of_batches)):
                # Here we are asking every worker to calculate their gradients and return
                # once they all calculate their gradients
                start_calculating_gradients_time = time.time()
                if circular:
                    ray.get(
                        [
                            worker.calculate_gradients_circular.remote(batch_no)
                            for worker in self.workers
                        ]
                    )
                else:
                    ray.get(
                        [
                            worker.calculate_gradients_linear.remote(batch_no)
                            for worker in self.workers
                        ]
                    )
                self.bolt_computation_time += (
                    time.time() - start_calculating_gradients_time
                )

                start_circular_communication_time = time.time()
                if circular:
                    ray.get(
                        self.primary_worker.subwork_circular_communication.remote(
                            batch_no
                        )
                    )
                else:
                    ray.get(
                        self.primary_worker.subwork_linear_communication.remote(
                            batch_no
                        )
                    )
                self.averaging_and_communication_time += (
                    time.time() - start_circular_communication_time
                )

                start_receiving_gradient_circular_time = time.time()
                if circular:
                    ray.get(
                        [
                            worker.receive_gradients_circular_communication.remote()
                            for worker in self.workers
                        ]
                    )
                else:
                    ray.get(
                        [
                            worker.receive_gradients_linear_communication.remote()
                            for worker in self.workers
                        ]
                    )
                self.averaging_and_communication_time += (
                    time.time() - start_receiving_gradient_circular_time
                )

                start_update_parameter_time = time.time()
                ray.get(
                    self.primary_worker.subwork_update_parameters.remote(
                        self.learning_rate
                    )
                )
                self.bolt_computation_time += time.time() - start_update_parameter_time

                self.logging.info(
                    "Epoch No: "
                    + str(epoch)
                    + ", Batch No: "
                    + str(batch_no)
                    + ", Bolt Computation Time: "
                    + str(self.bolt_computation_time)
                    + ", Averaging and Communcation Time: "
                    + str(self.averaging_and_communication_time)
                )

            for id, worker in enumerate(self.workers):
                acc, _ = ray.get(worker.predict.remote())
                self.logging.info(
                    "Accuracy on workers %d: %lf", id, acc["categorical_accuracy"]
                )

    def predict(self):
        """Calls network.predict() on worker of head node and returns the predictions.

        Returns:
            InferenceMetricData: Tuples of metrics and activations
        """

        assert len(self.workers) > 0, "No workers are initialized now."
        return ray.get(self.workers[0].predict.remote())
