import sys

try:
    from thirdai._distributed_bolt.backend.communication.circular import (
        Circular as Circular,
    )
except ImportError:
    pass

import pytest
import numpy as np


@pytest.mark.skipif("ray" not in sys.modules, reason="requires the ray library")
# @pytest.mark.xfail
def test_all_reduce_circular_communication():
    num_workers = 16
    circular_communicating_workers = [
        Circular(None, i, None, num_workers) for i in range(num_workers)
    ]

    for i in range(num_workers):
        circular_communicating_workers[i].set_friend(
            circular_communicating_workers[(i - 1) % num_workers]
        )

    weight_matrix_shapes = [(3, 60), (18,)]

    weights_all_reduced_gt = [np.zeros(shape) for shape in weight_matrix_shapes]
    # setting up gradients for each worker
    for i in range(num_workers):
        circular_communicating_workers[i].gradients = [
            np.random.randint(100, size=shape).astype("float32") for shape in weight_matrix_shapes
        ]
        for j in range(len(weight_matrix_shapes)):
            weights_all_reduced_gt[j] += circular_communicating_workers[i].gradients[j]
        circular_communicating_workers[i].calculate_gradient_partitions()

    for i in range(len(weights_all_reduced_gt)):
        weights_all_reduced_gt[i] /= num_workers

    for update_id, reduce in [(num_workers, True), (num_workers + 1, False)]:
        for node in range(num_workers - 1):
            should_avg_gradients = (node == num_workers - 2)
            for worker_id in range(num_workers):
                partition_id = (update_id + worker_id - 1) % num_workers
                worker = circular_communicating_workers[worker_id]
                worker.friend_gradients = worker.friend.receive_array_partitions(update_id)
                worker.update_partitions(partition_id, reduce=reduce, avg_gradients=should_avg_gradients)  
            update_id -= 1


    for worker in circular_communicating_workers:
        for gradient_id in range(len(weights_all_reduced_gt)):
            assert(np.array_equal(weights_all_reduced_gt[gradient_id], circular_communicating_workers[worker_id].gradients[gradient_id]))
