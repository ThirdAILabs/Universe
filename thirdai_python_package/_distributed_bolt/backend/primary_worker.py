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

        super().__init__(num_workers, 0, self, config, layer_dims, communication_type)

    def subwork_circular_communication(self, workers):
        """
        This function first call the workers to compute the gradients on their network
        and then implements Baidu's All Ring All Reduce algorithm for communication.
        Read more about that here:
        https://andrew.gibiansky.com/blog/machine-learning/baidu-allreduce/.

        :param workers: List of all the actor including primary worker
        :type workers: List[ray.actor]
        """

        # update_id imples here, the different stages of circular communication
        update_id = self.num_workers
        for node in range(self.num_workers - 1):
            if node == self.num_workers - 2:
                ray.get(
                    [
                        worker.process_ring.remote(update_id, avg_gradients=True)
                        for worker in workers
                    ]
                )
            else:
                ray.get([worker.process_ring.remote(update_id) for worker in workers])
            update_id -= 1

        # + 1, because it is the partition for the candidates giving the partitions
        update_id = self.num_workers + 1
        for node in range(self.num_workers - 1):
            ray.get(
                [
                    worker.process_ring.remote(update_id, reduce=False)
                    for worker in workers
                ]
            )
            update_id -= 1

    def subwork_linear_communication(self, workers):
        """
        This function implements the linear way of communicating between the node.
        In this way of communication, each of the worker calculates their gradients,
        send their gradients to the supervisor and the supervisor sums the gradients,
        averages it and and send the gradients back to the workers.

        :param workers: batch number for the particular worker with worker id (id).
        :type workers: int
        """
        gradients_list = ray.get(
            [worker.get_calculated_gradients.remote() for worker in workers]
        )

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
        self.w_gradients_avg = np.divide(self.w_gradients_avg, len(workers))
        self.b_gradients_avg = np.divide(self.b_gradients_avg, len(workers))

    def gradients_avg(self):
        """
        This function is called by the workers to get the gradients back from PrimaryWorker.
        Calling this function returns the averaged gradients which is already calculated
        by the PrimaryWorker.

        :return: returns tuple of weight gradient average and bias gradient average
        :rtype: Tuple[numpy.ndarray, numpy.ndarray]
        """
        return self.w_gradients_avg, self.b_gradients_avg

    def subwork_update_parameters(self, learning_rate: float, workers) -> bool:
        """
        This function calls every worker to update their parameters(weight and biases) with the
        updated gradients(which they receive from the PrimaryWorker)

        :param learning_rate: learning_rate for the training
        :type learning_rate: float
        :param workers: List of workers including primary worker
        :type workers: List[ray.worker]
        :return: Returns True on Completion
        :rtype: bool
        """
        ray.get([worker.update_parameters.remote(learning_rate) for worker in workers])
        return True

    def get_weights_biases(self):
        """
        This function is called by all the workers(other than worker with id = 0), here
            all the workers get the same initialized weights and bias as that of worker with id 0

        :return: return a list of weight and bias
        :rtype: Tuple[numpy.ndarray, numpy.ndarray]
        """
        self.weights_biases = self.return_params()
        return self.weights_biases
