import ray
import thirdai._distributed_bolt.backend.communication as communication
import time as time
from typing import Tuple, Any, Optional, Dict, List
import textwrap


class DistributedBolt:
    """Implements all the user level Distributed Bolt APIs."""

    def __init__(
        self,
        workers,
        logger,
        epochs,
        primary_worker,
        num_of_batches,
        communication_type,
    ):
        """Initializes the DistributeBolt class.

        Args:
            workers (List[Ray Actor]): Store all the workers including primary
            logger (Logging): gives the Logger
            epochs (int): number of epochs
            primary_worker (Ray Actor): Primary Worker
            num_of_batches (int): number of training batches
        """

        self.logger = logger
        self.workers = workers
        self.epochs = epochs
        self.num_of_batches = num_of_batches
        self.primary_worker = primary_worker
        self.communication_type = communication_type
        if self.communication_type not in communication.AVAILABLE_METHODS:
            raise ValueError(
                textwrap.dedent(
                    """
                        Currently only two modes of communication is supported.
                        Use: "circular" or "linear". 
                    """
                )
            )

    def train(self) -> None:
        """Trains the network using the communication type choosen.

        Args:
            circular (Optional[bool], optional): True, if circular communication is required.
                    False, if linear communication is required.. Defaults to True.
        """
        comm = (
            communication.Circular(self.workers, self.primary_worker, self.logging)
            if self.communication_type == "circular"
            else communication.Linear(self.workers, self.primary_worker, self.logging)
        )

        for epoch in range(self.epochs):
            for batch_id in range(self.num_of_batches):

                # Here we are asking every worker to calculate their gradients and return
                # once they all calculate their gradients
                comm.calculate_gradients(batch_id)
                comm.communicate()
                comm.update_parameters(self.learning_rate)
                comm.log_training(batch_id, epoch)

        ray.get([worker.finish_training.remote() for worker in self.workers])

    def predict(self):
        """Calls network.predict() on worker of head node and returns the predictions.

        Returns:
            InferenceMetricData: Tuples of metrics and activations
        """

        assert len(self.workers) > 0, "No workers are initialized now."
        return ray.get(self.workers[0].predict.remote())
