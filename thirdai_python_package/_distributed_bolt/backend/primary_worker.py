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
        self,
        layer_dims: List[int],
        no_of_workers: int,
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

        super().__init__(no_of_workers, 0, self)

    def set_workers(self, workers):
        self.workers = workers

    def subwork_circular_communication(self):
        """This function first call the workers to compute the gradients on their network
        and then implements Baidu's All Ring All Reduce algorithm for communication.
        Read more about that here:
        https://andrew.gibiansky.com/blog/machine-learning/baidu-allreduce/.

        Args:
            batch_no (int): batch number for the particular worker with worker id (id).

        Returns:
            _type_: _description_
        """

        update_id = 0
        for node in range(self.total_nodes - 1):
            if node == self.total_nodes - 2:
                ray.get(
                    [
                        worker.process_ring.remote(update_id, avg_gradients=True)
                        for worker in self.workers
                    ]
                )
            else:
                ray.get(
                    [worker.process_ring.remote(update_id) for worker in self.workers]
                )
            update_id -= 1

        update_id = 1
        for node in range(self.total_nodes - 1):
            ray.get(
                [
                    worker.process_ring.remote(update_id, reduce=False)
                    for worker in self.workers
                ]
            )
            update_id -= 1

    def subwork_linear_communication(self):
        """This function implements the linear way of communicating between the node.
        In this way of communication, each of the worker calculates their gradients,
        send their gradients to the supervisor and the supervisor sums the gradients,
        averages it and and send the gradients back to the workers.

        Args:
            batch_no (int): batch number for the particular worker with worker id (id).

        Returns:
            _type_: _description_
        """
        gradients_list = ray.get(
            [worker.get_calculated_gradients.remote() for worker in self.workers]
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
        self.w_gradients_avg = np.divide(self.w_gradients_avg, len(self.workers))
        self.b_gradients_avg = np.divide(self.b_gradients_avg, len(self.workers))

    def gradients_avg(self):
        """This function is called by the workers to get the gradients back from PrimaryWorker.
        Calling this function returns the averaged gradients which is already calculated
        by the PrimaryWorker.

        Returns:
            __type__: returns tuple of weight gradient average and bias gradient average
        """
        return self.w_gradients_avg, self.b_gradients_avg

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
