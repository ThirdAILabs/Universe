import ray
import time
from typing import Tuple, Any, Optional, Dict, List
from .Model import Model


class Worker:
    """
    This is a ray remote class(Actor). Read about them here.
    (https://docs.ray.io/en/latest/ray-core/actors.html)

    Worker is a ray actor which implements all the lower level
    functionalities between the Distributed Bolt APIs and
    Bolt native code.

    Args:
        layers: List of layer dimensions
        config: configuration file for setting up the network
        total_nodes: total number of nodes
        id: id of this particular worker
    """

    def __init__(self, layers: List, config, total_nodes: int, id: int):

        # Setting up Model
        self.Model = Model(config, total_nodes, layers, id)

        # Set up variables
        self.layers = layers
        self.total_nodes = total_nodes
        self.id = id

    def addHeadWorker(self, head_worker):
        """

        This function assigns each of the worker their head_worker
        """
        self.head_worker = head_worker

    def addFriend(self, friend):
        """

        This function is only needed for circular way of communication.
        This function assigns each of the worker their friend to which
        they will be communicating their gradients. Look at this link:
        https://andrew.gibiansky.com/blog/machine-learning/baidu-allreduce/
        """
        self.friend = friend

    def calculateGradientsCircular(self, batch_no: int):
        """

        This function is called only when the mode of
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
            batch_no: training batch to calculate gradients on.

        """
        start_calculate_grdient_time = time.time()
        self.Model.calculateGradients(batch_no)
        calculate_gradient_time = time.time() - start_calculate_grdient_time

        start_receive_gradients_time = time.time()
        self.w_partitions = []
        self.b_partitions = []

        self.w_gradients, self.b_gradients = self.Model.getCalculatedGradients()

        for x in self.w_gradients:
            self.w_partitions.append(int(len(x) / self.total_nodes))

        for y in self.b_gradients:
            self.b_partitions.append(int(len(y) / self.total_nodes))

        receive_gradients_time = time.time() - start_receive_gradients_time
        return calculate_gradient_time, receive_gradients_time

    def calculateGradientsLinear(self, batch_no: int):
        """
        This function is called only when the mode of communication is
        linear.

        This functions calls the API 'calculateGradientSingleNode',
        which calculates the gradients for the network managed by
        this particular worker. The calculateGradientSingleNode trains
        the network and calculates the gradient for the particular
        training batch with batch no. batch_no and with loss function
        specified in the config.

        Args:
            batch_no: training batch to calculate gradients on.
        """
        self.Model.calculateGradients(batch_no)
        return True

    def getCalculatedGradients(self):
        """
        This function is called only when the mode of communication
        is Linear.

        This function is called by the head_worker to compute the
        averages of the calculated gradients. This functions
        calls 'get_weights_gradient' and 'get_biases_gradients' functions
        inside bolt to take the gradients and return them to head_worker.
        """
        return self.Model.getCalculatedGradients()

    def returnParams(self):
        """

        This function will only be called for worker having its id 0.
        The head_worker will call this function to get the initial random
        weights from worker with id 0 and then send those weights to all
        the workers.

        """
        return self.Model.getParameters()

    def receiveParams(self):
        """

        This function is called by head_worker to all the workers whose id
        is not equal to 0. This function gets the initialized random weight
        ans biases from worker with id = 0. and sets the weight on all
        the other workers.

        """
        weights, biases = ray.get(self.head_worker.weights_biases.remote())
        self.Model.setParameters(weights, biases)
        return True

    def receiveGradientsCircularCommunication(self):
        """

        This function is called only when the communication pattern choosen
        is circular.

        This function is called by the head_worker to make set the updated
        gradients to the network.

        """
        self.Model.setGradients(self.w_gradients, self.b_gradients)
        return True

    def receiveGradientsLinearCommunication(self):
        """

        This function is called only when the communication pattern choosen
        is linear.

        This function is called by the head_worker to first, get the updated gradients
        from the head_worker and then set those updated gradients to the network.

        """

        self.w_gradients, self.b_gradients = ray.get(
            self.head_worker.gradients_avg.remote()
        )
        self.Model.setGradients(self.w_gradients, self.b_gradients)
        return True

    def calculate_partitions(
        self,
        partition_length: int,
        partition_id: int,
        total_length: int,
    ):
        l_idx = partition_length * partition_id
        r_idx = partition_length * (partition_id + 1)
        if total_length - r_idx < partition_length:
            r_idx = total_length
        return l_idx, r_idx

    def processRing(
        self,
        update_id: int,
        reduce: Optional[bool] = True,
        avg_gradients: Optional[bool] = False,
    ):
        """

        This function contains the main code for the circular ring communication
        pattern.

        The function first calculates the partition index range on which it will
        work, then get the graidnets on that range from its friend worker and sums
        it to the partition the partition the current worker.

        Here Each of the node communicates the partitioned gradients with
        their friend nodes, and those friend node communicate with their friends
        and the communication there by happens in a circle.


        Args:
            update_id: This id is use to calculate the partition to work on.
            reduce: This bool determines whether we need to reduce or gather
                True: redue, Flase: Gather
            avg_gradients: This bool determines whether we will average the
                gradients, or not. True: do averaging(i.e., divide by total_nodes),
                False: Do nothing

        """
        python_computation_time = 0
        communication_time = 0

        partition_id = (update_id + self.id - 1) % self.total_nodes

        t2 = time.time()
        get_ray_object = self.friend.receiveArrayPartitions.remote(update_id)
        (
            self.friend_weight_gradient_list,
            self.friend_bias_gradient_list,
            python_computation_time_receive_array,
        ) = ray.get(get_ray_object)
        communication_time += time.time() - t2 - python_computation_time_receive_array

        python_computation_time += python_computation_time_receive_array

        t2 = time.time()
        for i in range(len(self.friend_weight_gradient_list)):

            # Getting the indices of the partition to work on
            l_weight_idx, r_weight_idx = self.calculate_partitions(
                partition_length=self.w_partitions[i],
                partition_id=partition_id,
                total_length=len(self.w_gradients[i]),
            )
            l_bias_idx, r_bias_idx = self.calculate_partitions(
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

        python_computation_time += time.time() - t2
        return python_computation_time, communication_time

    def receiveArrayPartitions(self, update_id: int):
        """
        This function will only be get called for circular ring communication
        pattern.

        This function returns the array partition to the worker it is called by.


        Args:
            update_id: This id is use to calculate the partition to work on.
        """
        t1 = time.time()
        python_computation_time = 0
        partition_id = (update_id + self.id) % self.total_nodes

        w_gradient_subarray = []
        b_gradient_subarray = []
        for i in range(len(self.w_partitions)):

            # Getting the indices of the partition to work on
            l_weight_idx, r_weight_idx = self.calculate_partitions(
                partition_length=self.w_partitions[i],
                partition_id=partition_id,
                total_length=len(self.w_gradients[i]),
            )
            l_bias_idx, r_bias_idx = self.calculate_partitions(
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

        python_computation_time += time.time() - t1
        return w_gradient_subarray, b_gradient_subarray, python_computation_time

    def updateParameters(self, learning_rate: float):
        """

        This function calls updateParameter function inside bolt, which
        inherently updates the entire network.

        Args:
            learning_rate: the learning rate for updating the parameters
        """
        self.Model.updateParameters(learning_rate)
        return True

    def num_of_batches(self):
        """

        This function returns the total number of batches the workers have.
        """
        return self.Model.num_of_batches()

    def predict(self):
        """
        This function calls the predict function(predictSingleNode) to return the
        prediction from the network manges by this single worker.
        """
        return self.Model.predict()
