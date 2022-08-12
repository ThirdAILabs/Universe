import ray
import time
from typing import Tuple, Any, Optional, Dict, List
from .model import Model


def calculate_partitions(partition_length: int, partition_id: int, total_length: int):
    """This function returns the partitions for the work to work on,
    during the circular communication.

    Args:
        partition_length (int): length of partition to return
        partition_id (int): the partition id, which needed to be worked on
        total_length (int): length of the array to be transferred using
            circular communication

    Returns:
        Tuple[int,int]: Left Index and Right Index for a tuple
    """
    l_idx = partition_length * partition_id
    r_idx = partition_length * (partition_id + 1)
    if total_length - r_idx < partition_length:
        r_idx = total_length
    return l_idx, r_idx


class Worker:
    """This is a ray remote class(Actor). Read about them here.
    (https://docs.ray.io/en/latest/ray-core/actors.html)

    Worker is a ray actor which implements all the lower level
    functionalities between the Distributed Bolt APIs and
    Bolt native code.

    """

    def __init__(
        self, layer_dims: List, config: Dict, total_nodes: int, id: int, primary_worker
    ):
        """Initializes the model to run

        Args:
            layers (List): List of layer dimensions
            config (Dict): configuration file for setting up the network
            total_nodes (int): total number of nodes
            id (int): id of this particular worker
        """

        # Setting up Model
        self.model = Model(config, total_nodes, layer_dims, id)

        # Set up variables
        self.total_nodes = total_nodes
        self.id = id
        self.primary_worker = primary_worker

        # class variable for circular function
        self.friend = None  # this variable is set up in add_friend
        self.w_partitions = []
        self.b_partitions = []
        self.friend_bias_gradient_list = []
        self.friend_weight_gradient_list = []

    def add_friend(self, friend):
        """This function is only needed for circular way of communication.
        This function assigns each of the worker their friend to which
        they will be communicating their gradients. Look at this link:
        https://andrew.gibiansky.com/blog/machine-learning/baidu-allreduce/

        Args:
            friend (_type_): storing the friend for this worker
        """
        self.friend = friend

    def calculate_gradients_circular(self, batch_no: int):
        """This function is called only when the mode of
        communication is circular.


        This functions calls the API 'calculateGradientSingleNode',
        which calculates the gradients for the network managed by
        this particular worker. The calculateGradientSingleNode trains
        the network and calculates the gradient for the particular
        training batch with batch no. batch_no and with loss function
        specified in the config.

        This function also defines the partition size which defines the
        size of block of gradients which are communicated between a worker
        and its friend.

        Args:
            batch_no (int): training batch to calculate gradients on.

        Returns:
            _type_: _description_
        """
        self.model.calculate_gradients(batch_no)

        self.w_partitions = []
        self.b_partitions = []

        self.w_gradients, self.b_gradients = self.model.get_calculated_gradients()

        for x in self.w_gradients:
            self.w_partitions.append(int(len(x) / self.total_nodes))

        for y in self.b_gradients:
            self.b_partitions.append(int(len(y) / self.total_nodes))

    def calculate_gradients_linear(self, batch_no: int):
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
        self.model.calculate_gradients(batch_no)
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

    def receive_gradients_circular_communication(self) -> bool:
        """This function is called only when the communication pattern choosen
        is circular.

        This function is called by the primary_worker to make set the updated
        gradients to the network.

        Returns:
            bool: returns True, after functions complete
        """
        self.model.set_gradients(self.w_gradients, self.b_gradients)
        return True

    def receive_gradients_linear_communication(self) -> bool:
        """This function is called only when the communication pattern choosen
        is linear.

        This function is called by the primary_worker to first, get the updated gradients
        from the primary_worker and then set those updated gradients to the network.

        Returns:
            bool: returns True, after functions complete
        """
        if self.id is 0:
            self.w_gradients, self.b_gradients = self.primary_worker.gradients_avg()
        else:
            self.w_gradients, self.b_gradients = ray.get(
                self.primary_worker.gradients_avg.remote()
            )
        self.model.set_gradients(self.w_gradients, self.b_gradients)
        return True

    def process_ring(
        self,
        update_id: int,
        reduce: Optional[bool] = True,
        avg_gradients: Optional[bool] = False,
    ):
        """This function contains the main code for the circular ring communication
        pattern.

        The function first calculates the partition index range on which it will
        work, then get the graidnets on that range from its friend worker and sums
        it to the partition the partition the current worker.

        Here Each of the node communicates the partitioned gradients with
        their friend nodes, and those friend node communicate with their friends
        and the communication there by happens in a circle.

        Args:
            update_id (int): This id is use to calculate the partition to work on.
            reduce (Optional[bool], optional): This bool determines whether we need
            to reduce or gather, True: reduce, False: Gather. Defaults to True.
            avg_gradients (Optional[bool], optional): _description_. Defaults to False.

        Returns:
            _type_: _description_
        """

        partition_id = (update_id + self.id - 1) % self.total_nodes

        get_ray_object = self.friend.receive_array_partitions.remote(update_id)
        (
            self.friend_weight_gradient_list,
            self.friend_bias_gradient_list,
        ) = ray.get(get_ray_object)
        for i in range(len(self.friend_weight_gradient_list)):

            # Getting the indices of the partition to work on
            l_weight_idx, r_weight_idx = calculate_partitions(
                partition_length=self.w_partitions[i],
                partition_id=partition_id,
                total_length=len(self.w_gradients[i]),
            )
            l_bias_idx, r_bias_idx = calculate_partitions(
                partition_length=self.b_partitions[i],
                partition_id=partition_id,
                total_length=len(self.b_gradients[i]),
            )

            assert (
                self.w_partitions[i] > 0
            ), f"weight partions has value {self.w_partitions[i]}"
            assert (
                self.b_partitions[i] > 0
            ), f"bias partions has value {self.b_partitions[i]}"
            assert (
                r_weight_idx - l_weight_idx >= self.w_partitions[i]
            ), f"weight update index range are less than {self.w_partitions[i]}"
            assert (
                r_bias_idx - l_bias_idx >= self.b_partitions[i]
            ), f"bias update index range are less than {self.b_partitions[i]}"

            # arrays should be numpy arrays for the following operation, otherwise it will just get appened to the list
            if reduce:
                self.w_gradients[i][
                    l_weight_idx:r_weight_idx
                ] += self.friend_weight_gradient_list[i]
                self.b_gradients[i][
                    l_bias_idx:r_bias_idx
                ] += self.friend_bias_gradient_list[i]
                if avg_gradients:
                    self.w_gradients[i][l_weight_idx:r_weight_idx] = (
                        self.w_gradients[i][l_weight_idx:r_weight_idx]
                        / self.total_nodes
                    )
                    self.b_gradients[i][l_bias_idx:r_bias_idx] = (
                        self.b_gradients[i][l_bias_idx:r_bias_idx] / self.total_nodes
                    )
            else:
                self.w_gradients[i][
                    l_weight_idx:r_weight_idx
                ] = self.friend_weight_gradient_list[i]
                self.b_gradients[i][
                    l_bias_idx:r_bias_idx
                ] = self.friend_bias_gradient_list[i]

    def receive_array_partitions(self, update_id: int):
        """This function will only be get called for circular ring communication
        pattern.

        This function returns the array partition to the worker it is called by.

        Args:
            update_id (int): This id is use to calculate the partition to work on.

        Returns:
            _type_: _description_
        """
        partition_id = (update_id + self.id) % self.total_nodes

        w_gradient_subarray = []
        b_gradient_subarray = []
        for i in range(len(self.w_partitions)):

            # Getting the indices of the partition to work on
            l_weight_idx, r_weight_idx = calculate_partitions(
                partition_length=self.w_partitions[i],
                partition_id=partition_id,
                total_length=len(self.w_gradients[i]),
            )
            l_bias_idx, r_bias_idx = calculate_partitions(
                partition_length=self.b_partitions[i],
                partition_id=partition_id,
                total_length=len(self.b_gradients[i]),
            )

            assert (
                self.w_partitions[i] > 0
            ), f"weight partions has value {self.w_partitions[i]}"
            assert (
                self.b_partitions[i] > 0
            ), f"bias partions has value {self.b_partitions[i]}"
            assert (
                r_weight_idx - l_weight_idx >= self.w_partitions[i]
            ), f"weight update index range are less than {self.w_partitions[i]}"
            assert (
                r_bias_idx - l_bias_idx >= self.b_partitions[i]
            ), f"bias update index range are less than {self.b_partitions[i]}"

            w_gradient_subarray.append(self.w_gradients[i][l_weight_idx:r_weight_idx])
            b_gradient_subarray.append(self.b_gradients[i][l_bias_idx:r_bias_idx])

        return w_gradient_subarray, b_gradient_subarray

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

    def predict(self):
        """This function calls the predict function(predictSingleNode) to return the
        prediction from the network manges by this single worker.

        Returns:
            InferenceMetricData: Tuples for activations and metrics
        """
        return self.model.predict()
