import ray
from thirdai._distributed_bolt.backend.trainer import Trainer
import time as time
from typing import Tuple, Any, Optional, Dict, List
import textwrap
from thirdai._distributed_bolt.backend.communication import AVAILABLE_METHODS


class DistributedBolt:
    """
    Implements all the user level Distributed Bolt APIs
    """

    def __init__(
        self,
        workers,
        logger,
        epochs,
        primary_worker,
        num_of_batches,
        communication_type,
    ):
        """
        Initializes the DistributeBolt class.

        :param workers: Store all the workers including primary
        :type workers: [ray.actor]
        :param logger: gives the Logger
        :type logger: logging
        :param epochs: number of epochs
        :type epochs: int
        :param primary_worker: Primary Worker
        :type primary_worker: ray.actor
        :param num_of_batches: number of training batches
        :type num_of_batches: int
        :param communication_type: Type of Communication
        :type communication_type: string
        :raises ValueError: If communication method does not exist
        """

        self.logger = logger
        self.workers = workers
        self.epochs = epochs
        self.num_of_batches = num_of_batches
        self.primary_worker = primary_worker
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

    def train(self) -> None:
        """
        Trains the network using the communication type choosen.
        """
        trainer = Trainer(
            self.workers, self.primary_worker, self.logging, self.communication_type
        )

        for epoch in range(self.epochs):
            for batch_id in range(self.num_of_batches):

                # Here we are asking every worker to calculate their gradients and return
                # once they all calculate their gradients
                trainer.train(epoch, batch_id, self.learning_rate)

        trainer.finish_training()

    def predict(self):
        """
        Calls network.predict() on worker of head node and returns the predictions.

        :return: Tuples of metrics and activations
        :rtype: InferenceMetricData
        """

        assert len(self.workers) > 0, "No workers are initialized now."
        return ray.get(self.workers[0].predict.remote())
