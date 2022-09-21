import numpy as np
import ray
import time
from typing import Tuple, Any, Optional, Dict, List
from thirdai._distributed_bolt.backend.worker import Worker


@ray.remote(max_restarts=2)
class PrimaryWorker(Worker):
    """
    This is a ray remote class(Actor). Read about them here.
        (https://docs.ray.io/en/latest/ray-core/actors.html)

        PrimaryWorker is a ray actor which inherits all the function from
        Worker class. Apart from acting as a Worker, it also extends the worker
        class to implement functions to control the training. It controls
        training on each of the node(which batch number to train) and communication
        between the worker nodes.

    :param Worker: Inherits Worker Class
    :type Worker: ray.actor
    """

    def __init__(
        self, layer_dims: List[int], num_workers: int, config, communication_type
    ):
        """
        Initializes the Primary Worker Class

        :param layer_dims: List of layer dimensions.
        :type layer_dims: List[int]
        :param num_workers: number of workers in training
        :type num_workers: int
        :param config: configuration file dictionary
        :type config: TOML File
        :param communication_type: Type of Communication
        :type communication_type: string
        """
        self.layer_dims = layer_dims
        self.num_workers = num_workers
        super().__init__(num_workers, 0, self, config, layer_dims, communication_type)

    def average_aggregated_gradients(self, gradients_list_ref):
        """
        This function implements the linear way of communicating between the node.
        In this way of communication, each of the worker calculates their gradients,
        send their gradients to the trainer and here, the primary-worker sums the gradients,
        averages it

        :param gradients_list_ref: _description_
        :type gradients_list_ref: _type_
        """        
        gradients_list = ray.get(gradients_list_ref)

        # Here we are initializing the w_average_gradients for storing the sum
        self.w_gradients_avg = np.array(
            [
                np.zeros((self.layer_dims[layer_no + 1], self.layer_dims[layer_no]))
                for layer_no in range(len(self.layer_dims) - 1)
            ]
        )
        self.b_gradients_avg = np.array(
            [
                np.zeros((self.layer_dims[layer_no + 1]))
                for layer_no in range(len(self.layer_dims) - 1)
            ]
        )

        # summing all the gradients
        for w_gradients, b_gradients in gradients_list:
            self.w_gradients_avg += w_gradients
            self.b_gradients_avg += b_gradients

        # averaging the gradients
        self.w_gradients_avg = np.divide(self.w_gradients_avg, self.num_workers)
        self.b_gradients_avg = np.divide(self.b_gradients_avg, self.num_workers)

    def gradients_avg(self):
        """
        This function is called by the workers to get the gradients back from PrimaryWorker.
        Calling this function returns the averaged gradients which is already calculated
        by the PrimaryWorker.

        :return: returns tuple of weight gradient average and bias gradient average
        :rtype: Tuple[numpy.ndarray, numpy.ndarray]
        """
        return self.w_gradients_avg, self.b_gradients_avg

    def get_weights_biases(self):
        """
        This function is called by all the workers(other than worker with id = 0), here
            all the workers get the same initialized weights and bias as that of worker with id 0

        :return: return a list of weight and bias
        :rtype: Tuple[numpy.ndarray, numpy.ndarray]
        """
        self.weights_biases = self.return_params()
        return self.weights_biases
