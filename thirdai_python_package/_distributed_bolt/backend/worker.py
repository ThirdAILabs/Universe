from numpy import partition
import ray
import time
from typing import Tuple, Any, Optional, Dict, List
from thirdai._distributed_bolt._models.fully_connected_network_model import (
    FullyConnectedNetworkSingleNode,
)
import thirdai._distributed_bolt.backend.communication as comm


class Worker:
    """
    This is a ray remote class(Actor). Read about them here.
    (https://docs.ray.io/en/latest/ray-core/actors.html)

    Worker is a ray actor which implements all the lower level
    functionalities between the Distributed Bolt APIs and
    Bolt native code.
    """

    def __init__(
        self,
        num_workers: int,
        id: int,
        primary_worker,
        config,
        layer_dims,
        communication_type,
    ):
        """
        Initializes the worker to run

        :param num_workers: total number of nodes
        :type num_workers: int
        :param id: id of this particular worker
        :type id: int
        :param primary_worker: Primary Worker
        :type primary_worker: ray.actor
        :param config: configuration file for setting up the network
        :type config: Dict
        :param layer_dims: dimensions for network
        :type layer_dims: List[int]
        :param communication_type: type of communication
        :type communication_type: string
        """

        self.model = FullyConnectedNetworkSingleNode(
            config, num_workers, layer_dims, id
        )
        # Set up variables
        self.num_workers = num_workers
        self.id = id
        self.primary_worker = primary_worker
        self.communication_type = communication_type

        self.comm = (
            comm.Circular(self.model, self.id, self.primary_worker, self.num_workers)
            if self.communication_type == "circular"
            else comm.Linear(self.model, self.id, self.primary_worker)
        )

    # see https://github.com/ray-project/ray/blob/4b59dfbe59a143ab8dcc505dad860b4c330b6426/python/ray/actor.py#L1183
    # It looks like ray doesnot support direct class attribute access in python.
    # Hence, we will need to expose this function here in worker
    def set_friend(self, friend):
        """
        Add the friend for communicating for cicrcular all reduce

        :param friend: worker to which self need to communication
                            for circular all reduce
        :type friend: ray.actor
        """
        self.comm.set_friend(friend)

    def process_ring(
        self,
        update_id: int,
        reduce: Optional[bool] = True,
        avg_gradients: Optional[bool] = False,
    ):
        """
        This function handles the circular all reduce

        :param update_id: The update sequence id
        :type update_id: int
        :param reduce: True if reduce, False if gather, defaults to True
        :type reduce: Optional[bool], optional
        :param avg_gradients: whether the update requires updating the gradients, defaults to False
        :type avg_gradients: Optional[bool], optional
        """
        self.comm.process_ring(update_id, reduce, avg_gradients)

    def receive_array_partitions(self, update_id: int):
        """
        This function returns the array partition for the worker is is called.

        :param update_id: The update sequence id
        :type update_id: int
        :return: subarray partition
        :rtype: numpy.ndarray
        """
        return self.comm.receive_array_partitions(update_id)

    def calculate_gradients(self, batch_no: int):
        """
        This function is called only when the mode of communication is
        linear.

        This functions calls the API 'calculateGradientSingleNode',
        which calculates the gradients for the network managed by
        this particular worker. The calculateGradientSingleNode trains
        the network and calculates the gradient for the particular
        training batch with batch no. batch_no and with loss function
        specified in the config.

        :param batch_no: training batch to calculate gradients on.
        :type batch_no: int
        :return: check whether training is complete or not
        :rtype: bool
        """
        self.comm.calculate_gradients(batch_no)
        return True

    def get_calculated_gradients(self):
        """
        This function is called only when the mode of communication
        is Linear.

        This function is called by the primary_worker to compute the
        averages of the calculated gradients. This functions
        calls 'get_weights_gradient' and 'get_biases_gradients' functions
        inside bolt to take the gradients and return them to primary_worker.

        :return: Model Gradients
        :rtype: numpy.ndarray
        """
        return self.model.get_calculated_gradients()

    def return_params(self):
        """
        This function will only be called for worker having its id 0.
        The primary_worker will call this function to get the initial random
        weights from worker with id 0 and then send those weights to all
        the workers.

        :return: Model Parameters
        :rtype: numpy.ndarray
        """
        return self.model.get_parameters()

    def synchronize_parameters(self) -> bool:
        """
        This function is called by primary_worker to all the workers whose id
        is not equal to 0. This function gets the initialized random weight
        ans biases from worker with id = 0. and sets the weight on all
        the other workers.

        :return: signals the synchronization is complete
        :rtype: bool
        """
        if self.id != 0:
            weights, biases = ray.get(self.primary_worker.get_weights_biases.remote())
            self.model.set_parameters(weights, biases)
        return True

    def receive_gradients(self) -> bool:
        """
        This function is called only when the communication pattern choosen
        is circular.

        This function is called by the primary_worker to make set the updated
        gradients to the network.

        :return: receive updated gradients
        :rtype: bool
        """
        self.comm.receive_gradients()
        return True

    def update_parameters(self, learning_rate: float) -> bool:
        """
        This function calls updateParameter function inside bolt, which
        inherently updates the entire network.

        :param learning_rate: the learning rate for updating the parameters
        :type learning_rate: float
        :return: Returns true if function completes successfully
        :rtype: bool
        """
        self.model.update_parameters(learning_rate)
        return True

    def num_of_batches(self) -> int:
        """
        This function returns the total number of batches the workers have.
        """
        return self.model.num_of_batches()

    def finish_training(self):
        self.model.finish_training()

    def predict(self):
        return self.model.predict()
