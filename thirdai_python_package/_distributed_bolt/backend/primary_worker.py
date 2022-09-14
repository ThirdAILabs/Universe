import numpy as np
import ray
import time
from typing import Tuple, Any, Optional, Dict, List
from thirdai._distributed_bolt.backend.worker import Worker


@ray.remote(max_restarts=2)
class PrimaryWorker(Worker):
    """This is a ray remote class(Actor). Read about them here.
        (https://docs.ray.io/en/latest/ray-core/actors.html)

        PrimaryWorker is a ray actor which inherits all the function from
        Worker class. Apart from acting as a Worker, it also extends the worker
        class to implement functions to control the training. It controls
        training on each of the node(which batch number to train) and communication
        between the worker nodes.

    Args:
        Worker(Worker Class): Inherits Worker Class
    """

    def __init__(
        self, layer_dims: List[int], no_of_workers: int, config, communication_type
    ):
        """Initializes the Primary Worker Class

        Args:
            layers (List[int]): List of layer dimensions.
            config (Dict):  configuration file dictionary
            no_of_workers (int): number of workers in training
        """
        self.layer_dims = layer_dims

        # set up in add workers
        self.workers = None

        super().__init__(no_of_workers, 0, self, config, layer_dims, communication_type)

    def communicate(self):
        """The function calls comm to start communicating the gradients."""
        self.comm.communicate()

    def gradients_avg(self):
        """This function is called by the workers to get the gradients back from PrimaryWorker.
        Calling this function returns the averaged gradients which is already calculated
        by the PrimaryWorker.

        Returns:
            __type__: returns tuple of weight gradient average and bias gradient average
        """
        return self.comm.w_gradients_avg, self.comm.b_gradients_avg

    def subwork_update_parameters(self, learning_rate: float) -> bool:
        """This function calls every worker to update their parameters(weight and biases) with the
        updated gradients(which they receive from the PrimaryWorker)

        Args:
            learning_rate (float): learning_rate for the training

        Returns:
            bool: Returns True on Completion
        """
        ray.get(
            [worker.update_parameters.remote(learning_rate) for worker in self.workers]
        )
        return True

    def get_weights_biases(self):
        """This function is called by all the workers(other than worker with id = 0), here
            all the workers get the same initialized weights and bias as that of worker with id 0

        Returns:
            __type__: return a list of weight and bias
        """
        self.weights_biases = self.return_params()
        return self.weights_biases
