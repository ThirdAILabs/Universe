import ray
from typing import Tuple, Any, Optional, Dict, List


class Circular:
    def __init__(self, model, id, primary_worker, num_workers):

        self.model = model
        self.id = id
        self.primary_worker = primary_worker
        self.num_workers = num_workers

        self.friend = None  # this variable is set up in set_friend
        self.w_partitions = []
        self.b_partitions = []
        self.friend_bias_gradient_list = []
        self.friend_weight_gradient_list = []
        self.w_gradients = []
        self.b_gradients = []

    def set_friend(self, friend):
        """
        This function assigns each of the worker their friend to which
        they will be communicating their gradients. Look at this link:
        https://andrew.gibiansky.com/blog/machine-learning/baidu-allreduce/

        :param friend: storing the friend for this worker
        :type friend: ray.actor
        """
        self.friend = friend

    def calculate_gradients_partitions(self):
        """
        Calculate the partitions for distributed training called only
        in case of circular communication
        """
        for w_gradients in self.w_gradients:
            partition_length = int(len(w_gradients) / self.num_workers)
            remaining_length = len(w_gradients) % self.num_workers
            partition_start_end_list = []
            current_index = 0
            for i in range(self.num_workers):
                if i < remaining_length:
                    partition_start_end_list.append(
                        (current_index, current_index + partition_length + 1)
                    )
                    current_index += partition_length + 1
                else:
                    partition_start_end_list.append(
                        (current_index, current_index + partition_length)
                    )
                    current_index += partition_length

            self.w_partitions.append(partition_start_end_list)

        for b_layers in self.b_gradients:
            partition_length = int(len(b_layers) / self.num_workers)
            remaining_length = len(b_layers) % self.num_workers
            partition_start_end_list = []
            current_index = 0
            for i in range(self.num_workers):
                if i < remaining_length:
                    partition_start_end_list.append(
                        (current_index, current_index + partition_length + 1)
                    )
                    current_index += partition_length + 1
                else:
                    partition_start_end_list.append(
                        (current_index, current_index + partition_length)
                    )
                    current_index += partition_length

            self.b_partitions.append(partition_start_end_list)

    def calculate_gradients(self, batch_no: int):
        """
        This functions calls the API 'calculateGradientSingleNode',
        which calculates the gradients for the network managed by
        this particular worker. The calculateGradientSingleNode trains
        the network and calculates the gradient for the particular
        training batch with batch no. batch_no and with loss function
        specified in the config.

        This function also defines the partition size which defines the
        size of block of gradients which are communicated between a worker
        and its friend.

        :param batch_no: training batch to calculate gradients on
        :type batch_no: int
        """
        self.model.calculate_gradients(batch_no)

        self.w_partitions = []
        self.b_partitions = []

        self.w_gradients, self.b_gradients = self.model.get_calculated_gradients()

        self.calculate_gradients_partitions()

    def receive_gradients(self) -> bool:
        """
        This function is called by the primary_worker to make set the updated
        gradients to the network.

        :return: returns True, after functions complete
        :rtype: bool
        """
        self.model.set_gradients(self.w_gradients, self.b_gradients)
        return True

    def update_partitions(
        self,
        partition_id,
        reduce,
        avg_gradients,
    ):
        """
        Update the partitions with the partitioned array received from its friend

        :param partition_id: Partition index for partition to be updated
        :type partition_id: int
        :param reduce: This bool determines whether we need
                        to reduce or gather, True: reduce, False: Gather. Defaults to True.
        :type reduce: Optional[bool], optional
        :param avg_gradients: Defaults to False.
        :type avg_gradients: Optional[bool], optional
        """
        for i in range(len(self.friend_weight_gradient_list)):

            # Getting the indices of the partition to work on
            l_weight_idx, r_weight_idx = self.w_partitions[i][partition_id]
            l_bias_idx, r_bias_idx = self.b_partitions[i][partition_id]

            if r_weight_idx > l_weight_idx:

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
                            / self.num_workers
                        )
                        self.b_gradients[i][l_bias_idx:r_bias_idx] = (
                            self.b_gradients[i][l_bias_idx:r_bias_idx]
                            / self.num_workers
                        )
                else:
                    self.w_gradients[i][
                        l_weight_idx:r_weight_idx
                    ] = self.friend_weight_gradient_list[i]
                    self.b_gradients[i][
                        l_bias_idx:r_bias_idx
                    ] = self.friend_bias_gradient_list[i]

    def process_ring(
        self,
        update_id: int,
        reduce: Optional[bool] = True,
        avg_gradients: Optional[bool] = False,
    ):
        """
        The function first calculates the partition index range on which it will
        work, then get the gradients on that range from its friend worker and sums
        it to the partition the partition the current worker.

        Here each of the node communicates the partitioned gradients with
        their friend nodes, and those friend node communicate with their friends
        and the communication there by happens in a circle.

        :param update_id: This id is use to calculate the partition to work on.
        :type update_id: int
        :param reduce: This bool determines whether we need,
                    to reduce or gather, True: reduce, False: Gather. defaults to True
        :type reduce: Optional[bool], optional
        :param avg_gradients: _description_, defaults to False
        :type avg_gradients: Optional[bool], optional
        """

        partition_id = (update_id + self.id - 1) % self.num_workers

        get_ray_object = self.friend.receive_array_partitions.remote(update_id)
        (
            self.friend_weight_gradient_list,
            self.friend_bias_gradient_list,
        ) = ray.get(get_ray_object)
        self.update_partitions(partition_id, reduce, avg_gradients)

    def receive_array_partitions(self, update_id: int):
        """
        This function returns the array partition to the worker it is called by.

        :param update_id: This id is use to calculate the partition to work on.
        :type update_id: int
        :return: gradients subarray
        :rtype: numpy.ndarray
        """
        partition_id = (update_id + self.id) % self.num_workers

        w_gradient_subarray = []
        b_gradient_subarray = []
        for i in range(len(self.w_partitions)):

            # Getting the indices of the partition to work on
            l_weight_idx, r_weight_idx = self.w_partitions[i][partition_id]
            l_bias_idx, r_bias_idx = self.b_partitions[i][partition_id]

            if r_weight_idx > l_weight_idx:
                w_gradient_subarray.append(
                    self.w_gradients[i][l_weight_idx:r_weight_idx]
                )
                b_gradient_subarray.append(self.b_gradients[i][l_bias_idx:r_bias_idx])

        return w_gradient_subarray, b_gradient_subarray
