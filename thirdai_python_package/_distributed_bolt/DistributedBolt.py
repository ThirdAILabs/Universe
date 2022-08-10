import ray
import os
import toml
import textwrap
from .primary_worker import PrimaryWorker
from .replica_worker import ReplicaWorker
import time as time
from .utils import init_logging
from typing import Tuple, Any, Optional, Dict, List


class DistributedBolt:

    def __init__(
        self,
        no_of_workers: int,
        config_filename: str,
        num_cpus_per_node: Optional[int] = -1,
    ):
        """Implements all the user level Distributed Bolt APIs to the users.

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
            self.logging.error("Could not load the toml file! " + 'Config File Location:' + config_filename)

        if len(config["dataset"]["train_data"]) != no_of_workers:            
            raise ValueError("Received ", str(len(config["dataset"]["train_data"])) ," training datasets. Expected ", no_of_workers," datasets, one for each node.")

        self.no_of_workers = no_of_workers

        self.logging.info('Setting OMP_NUM_THREADS to ' + str(self.get_num_cpus()))
        runtime_env = {"env_vars": {"OMP_NUM_THREADS": str(self.get_num_cpus())}}

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
        self.layers = [config["dataset"]["input_dim"]]

        for i in range(len(config["layers"])):
            self.layers.append(config["layers"][i]["dim"])

        num_cpus = self.get_num_cpus()
        if num_cpus_per_node is not -1:
            num_cpus = num_cpus_per_node
        self.workers = [
            ReplicaWorker.options(num_cpus=num_cpus, max_concurrency=100).remote(
                self.layers, config, self.no_of_workers, id + 1
            )
            for id in range(self.no_of_workers - 1)
        ]
        self.head_worker = PrimaryWorker.options(
            num_cpus=num_cpus, max_concurrency=100
        ).remote(self.layers, config, self.no_of_workers)

        self.workers.insert(0, self.head_worker)
        self.head_worker.add_workers.remote(self.workers)

        self.num_of_batches = min(
            ray.get(
                [
                    worker.num_of_batches.remote()
                    for worker in self.workers
                ]
            )
        )

        for i in range(len(self.workers)):
            ray.get(self.workers[i].add_head_worker.remote(self.head_worker))
            ray.get(
                self.workers[i].add_friend.remote(
                    self.workers[(i - 1) % (len(self.workers))]
                )
            )

        self.bolt_computation_time = 0
        self.python_computation_time = 0
        self.communication_time = 0

    def get_num_cpus(self):
        """Returns the number of CPUs present on the machine

        Returns:
            _type_: None
        """
        try:
            import multiprocessing

            return multiprocessing.cpu_count()
        except (ImportError):
            print("Could not find num_cpus, setting num_cpus to DEFAULT=1")
            return 1

    def train(self, circular: Optional[bool] = True) -> None:
        """
        Trains the network using the communication type choosen.


        Args:
            circular: True, if circular communication is required.
                    False, if linear communication is required.
        """

        if circular:
            self.logging.info("Circular communication pattern is choosen")
            ray.get(
                [
                    worker.receive_params.remote()
                    for worker in self.workers
                ]
            )
            for epoch in range(self.epochs):

                for batch_no in range(int(self.num_of_batches)):

                    (
                        gradient_computation_time,
                        getting_gradient_time,
                        summing_and_averaging_gradients_time,
                    ) = ray.get(
                        self.head_worker.subwork_circular_communication.remote(batch_no)
                    )

                    start_gradients_send_time = time.time()
                    ray.get(
                        [
                            worker.receive_gradients_circular_communication.remote()
                            for worker in self.workers
                        ]
                    )
                    gradient_send_time = time.time() - start_gradients_send_time

                    start_update_parameters_time = time.time()
                    ray.get(
                        self.head_worker.subwork_update_parameters.remote(
                            self.learning_rate
                        )
                    )
                    update_parameter_time = time.time() - start_update_parameters_time

                    self.bolt_computation_time += (
                        gradient_computation_time + update_parameter_time
                    )
                    self.python_computation_time += summing_and_averaging_gradients_time
                    self.communication_time += (
                        getting_gradient_time + gradient_send_time
                    )

                    self.logging.info(
                        "Epoch No: "
                        + str(epoch)
                        + ", Batch No: "
                        + str(batch_no)
                        + ", Bolt Computation Time: "
                        + str(self.bolt_computation_time)
                        + ", Python Computation Time: "
                        + str(self.python_computation_time)
                        + ", Communication Time: "
                        + str(self.communication_time)
                    )

                for worker in self.workers:
                    acc, _ = ray.get(worker.predict.remote())
                    self.logging.info(
                        "Accuracy on workers %d: %lf", i, acc["categorical_accuracy"]
                    )
        else:
            self.logging.info("Linear communication pattern is choosen")

            updateWeightsAndBiases = ray.get(
                [
                    worker.receive_params.remote()
                    for worker in self.workers
                ]
            )

            for epoch in range(self.epochs):

                for batch_no in range(self.num_of_batches):

                    (
                        gradient_computation_time,
                        getting_gradient_time,
                        summing_and_averaging_gradients_time,
                    ) = ray.get(
                        self.head_worker.subwork_linear_communication.remote(batch_no)
                    )

                    start_gradients_send_time = time.time()
                    ray.get(
                        [
                            worker.receive_gradients_linear_communication.remote()
                            for worker in self.workers
                        ]
                    )
                    gradient_send_time = time.time() - start_gradients_send_time

                    start_update_parameters_time = time.time()
                    ray.get(
                        self.head_worker.subwork_update_parameters.remote(
                            self.learning_rate
                        )
                    )
                    update_parameters_time = time.time() - start_update_parameters_time

                    self.bolt_computation_time += (
                        gradient_computation_time + update_parameters_time
                    )
                    self.python_computation_time += summing_and_averaging_gradients_time
                    self.communication_time += (
                        getting_gradient_time + gradient_send_time
                    )

                    self.logging.info(
                        "Epoch No: "
                        + str(epoch)
                        + ", Batch No: "
                        + str(batch_no)
                        + ", Bolt Computation Time: "
                        + str(self.bolt_computation_time)
                        + ", Python Computation Time: "
                        + str(self.python_computation_time)
                        + ", Communication Time: "
                        + str(self.communication_time)
                    )

                for worker in self.workers:
                    acc, _ = ray.get(worker.predict.remote())
                    self.logging.info(
                        "Accuracy on workers %d: %lf", i, acc["categorical_accuracy"]
                    )

    def predict(self):
        """
        Calls network.predict() on one of worker on head node and returns the predictions.
        """

        assert len(self.workers) > 0, "No workers are initialized now."
        return ray.get(self.workers[0].predict.remote())
