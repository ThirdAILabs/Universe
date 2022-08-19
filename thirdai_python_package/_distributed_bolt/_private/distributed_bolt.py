import ray
from thirdai._distributed_bolt._private.primary_worker import PrimaryWorker
from thirdai._distributed_bolt._private.replica_worker import ReplicaWorker
import time as time
from typing import Tuple, Any, Optional, Dict, List


class DistributedBolt:
    """Implements all the user level Distributed Bolt APIs to the users."""

    def __init__(
        self, workers, logger, epochs, primary_worker, num_of_batches, model_type
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

        self.logger = logger
        self.workers = workers
        self.epochs = epochs
        self.num_of_batches = num_of_batches
        self.primary_worker = primary_worker
        self.model_type = model_type

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
