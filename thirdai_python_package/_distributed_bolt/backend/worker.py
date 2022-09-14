from numpy import partition
import ray
import time
from typing import Tuple, Any, Optional, Dict, List
from thirdai._distributed_bolt._models.fully_connected_network_model import (
    FullyConnectedNetworkSingleNode,
)
import thirdai._distributed_bolt.backend._communication as comm


class Worker:
    """This is a ray remote class(Actor). Read about them here.
    (https://docs.ray.io/en/latest/ray-core/actors.html)

    Worker is a ray actor which implements all the lower level
    functionalities between the Distributed Bolt APIs and
    Bolt native code.

    """

    def __init__(
        self,
        total_nodes: int,
        id: int,
        primary_worker,
        config,
        layer_dims,
        communication_type,
    ):
        """Initializes the model to run

        Args:
            layers (List): List of layer dimensions
            config (Dict): configuration file for setting up the network
            total_nodes (int): total number of nodes
            id (int): id of this particular worker
            config: Training Config File
            layer_dims: dimensions for network
            communication_type: type of communication worker gonna use
        """

        self.model = FullyConnectedNetworkSingleNode(
            config, total_nodes, layer_dims, id
        )
        # Set up variables
        self.total_nodes = total_nodes
        self.id = id
        self.primary_worker = primary_worker
        self.communication_type = communication_type

        self.comm = (
            comm.Circular(self.model, self.id, self.primary_worker, self.total_nodes)
            if self.communication_type == "circular"
            else comm.Linear(self.model, self.id, self.primary_worker)
        )

    # see https://github.com/ray-project/ray/blob/4b59dfbe59a143ab8dcc505dad860b4c330b6426/python/ray/actor.py#L1183
    # It looks like ray doesnot support direct class attribute access in python.
    # Hence, we will need to expose this function here in worker
    def set_friend(self, friend):
        """Add the friend for communicating for cicrcular all reduce

        Args:
            friend (Ray Actor Class): worker to which self need to communication
                            for circular all reduce
        """
        self.comm.set_friend(friend)

    def process_ring(
        self,
        update_id: int,
        reduce: Optional[bool] = True,
        avg_gradients: Optional[bool] = False,
    ):
        """This function handles the circular all reduce

        Args:
            update_id (int): The update sequence id.
            reduce (Optional[bool], optional): True if reduce, False if gather. Defaults to True.
            avg_gradients (Optional[bool], optional): whether the update requires updating the gradients.
                            Defaults to False.

        """
        self.comm.process_ring(update_id, reduce, avg_gradients)

    def receive_array_partitions(self, update_id: int):
        """This function returns the array partition for the worker is is called.

        Args:
            update_id (int): The update sequence id.

        Returns:
            _type_: _description_
        """
        return self.comm.receive_array_partitions(update_id)

    def calculate_gradients(self, batch_no: int):
        """This function is called only when the mode of communication is
        linear.

        This functions calls the API 'calculateGradientSingleNode',
        which calculates the gradients for the network managed by
        this particular worker. The calculateGradientSingleNode trains
        the network and calculates the gradient for the particular
        training batch with batch no. batch_no and with loss function
        specified in the config.

        Args:
            batch_no (int): training batch to calculate gradients on.

        Returns:
            _type_: _description_
        """
        self.comm.calculate_gradients(batch_no)
        return True

    def get_calculated_gradients(self):
        """This function is called only when the mode of communication
        is Linear.

        This function is called by the primary_worker to compute the
        averages of the calculated gradients. This functions
        calls 'get_weights_gradient' and 'get_biases_gradients' functions
        inside bolt to take the gradients and return them to primary_worker.

        Returns:
            _type_: _description_
        """
        return self.model.get_calculated_gradients()

    def return_params(self):
        """This function will only be called for worker having its id 0.
        The primary_worker will call this function to get the initial random
        weights from worker with id 0 and then send those weights to all
        the workers.

        Returns:
            _type_: _description_
        """
        return self.model.get_parameters()

    def synchronize_parameters(self) -> bool:
        """This function is called by primary_worker to all the workers whose id
        is not equal to 0. This function gets the initialized random weight
        ans biases from worker with id = 0. and sets the weight on all
        the other workers.

        Returns:
            bool: returns True, after functions complete
        """
        if self.id is 0:
            weights, biases = self.primary_worker.get_weights_biases()
        else:
            weights, biases = ray.get(self.primary_worker.get_weights_biases.remote())
        self.model.set_parameters(weights, biases)
        return True

    def receive_gradients(self) -> bool:
        """This function is called only when the communication pattern choosen
        is circular.

        This function is called by the primary_worker to make set the updated
        gradients to the network.

        Returns:
            bool: returns True, after functions complete
        """
        self.comm.receive_gradients()
        return True

    def update_parameters(self, learning_rate: float) -> bool:
        """This function calls updateParameter function inside bolt, which
        inherently updates the entire network.

        Args:
            learning_rate (float): the learning rate for updating the parameters

        Returns:
            bool: Returns true if function completes successfully
        """
        self.model.update_parameters(learning_rate)
        return True

    def num_of_batches(self) -> int:
        """This function returns the total number of batches the workers have.

        Returns:
            int: number of batches for training on this node
        """
        """
        This function returns the total number of batches the workers have.
        """
        return self.model.num_of_batches()

    def finish_training(self):
        self.model.finish_training()

    def predict(self):
        """This function calls the predict function(predictSingleNode) to return the
        prediction from the network manages by this single worker.

        Returns:
            InferenceMetricData: Tuples for activations and metrics
        """
        return self.model.predict()
