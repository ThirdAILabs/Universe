from typing import Optional

import numpy as np
import ray


class Circular:
    def __init__(self, model, id, primary_worker, num_workers):
        self.model = model
        self.id = id
        self.primary_worker = primary_worker
        self.num_workers = num_workers

        self.friend = None  # this variable is set up in set_friend
        self.partitions = []
        self.friend_gradients = []
        self.gradients = []

    def set_friend(self, friend):
        """
        This function assigns each of the worker their friend to which
        they will be communicating their gradients. Look at this link:
        https://andrew.gibiansky.com/blog/machine-learning/baidu-allreduce/

        :param friend: storing the friend for this worker
        :type friend: ray.actor
        """
        self.friend = friend

    def calculate_gradient_partitions(self):
        partition_length = int(len(self.gradients) / self.num_workers)
        remaining_length = len(self.gradients) % self.num_workers
        self.partitions = []
        last_partition_end_index = 0
        for worker_id in range(self.num_workers):
            if worker_id < remaining_length:
                self.partitions.append(
                    (
                        last_partition_end_index,
                        last_partition_end_index + partition_length + 1,
                    )
                )
                last_partition_end_index += partition_length + 1
            else:
                self.partitions.append(
                    (
                        last_partition_end_index,
                        last_partition_end_index + partition_length,
                    )
                )
                last_partition_end_index += partition_length

    def compute_and_store_batch_gradients(self, batch_id: int):
        """
        This functions calls the API 'compute_and_store_batch_gradients',
        which calculates the gradients for the network managed by
        this particular worker. The compute_and_store_batch_gradients trains
        the network and calculates the gradient for the particular
        training batch with batch no. batch_id and with loss function
        specified in the config.

        This function also defines the partition size which defines the
        size of block of gradients which are communicated between a worker
        and its friend.

        :param batch_id: training batch to calculate gradients on
        :type batch_id: int
        """
        self.model.compute_and_store_batch_gradients(batch_id)

        self.partitions = []
        self.gradients = np.array(self.model.gradient_reference().get_gradients())

        self.calculate_gradient_partitions()

    def receive_gradients(self):
        """
        This function is called by the primary_worker to set the updated
        gradients to the network (after the circular communication has
        finished).

        :return: returns True, after functions complete
        :rtype: bool
        """
        self.model.gradient_reference().set_gradients(self.gradients)

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
        :type reduce: bool, optional
        :param avg_gradients: Defaults to False.
        :type avg_gradients: bool, optional
        """

        # Getting the indices of the partition to work on
        start_row_index, end_row_index = self.partitions[partition_id]

        if start_row_index < end_row_index:
            # arrays should be numpy arrays for the following operation, otherwise it will just get appened to the list
            if reduce:
                self.gradients[start_row_index:end_row_index] += self.friend_gradients
                if avg_gradients:
                    self.gradients[start_row_index:end_row_index] /= self.num_workers
            else:
                self.gradients[start_row_index:end_row_index] = self.friend_gradients

    def process_ring(
        self,
        update_id: int,
        reduce: bool = True,
        avg_gradients: bool = False,
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
        :type reduce: bool
        :param avg_gradients: _description_, defaults to False
        :type avg_gradients: bool
        """

        partition_id = (update_id + self.id - 1) % self.num_workers

        friend_gradients_ref = self.friend.receive_array_partitions.remote(update_id)
        self.friend_gradients = ray.get(friend_gradients_ref)
        self.update_partitions(partition_id, reduce, avg_gradients)
        del friend_gradients_ref

    def receive_array_partitions(self, update_id: int):
        """
        This function returns the array partition to the worker it is called by.

        :param update_id: This id is use to calculate the partition to work on.
        :type update_id: int
        :return: gradients subarray
        :rtype: numpy.ndarray
        """
        partition_id = (update_id + self.id) % self.num_workers
        (start_row_index, end_row_index) = self.partitions[partition_id]

        send_gradients = []
        if start_row_index < end_row_index:
            send_gradients = self.gradients[start_row_index:end_row_index]
        else:
            # This won't happen in most use cases since the number
            # of parameters in the array would have to be less than the
            # number of nodes (assuming even partitions), but we include
            # it to handle the edge case gracefully.
            send_gradients = None

        return send_gradients
