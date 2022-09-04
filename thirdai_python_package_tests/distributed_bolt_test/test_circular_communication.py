try:
    from thirdai._distributed_bolt.backend.worker import Worker
except ImportError:
    import warnings

    warnings.warn(
        "Error while importing thirdai.distributed_bolt. "
        "You might be missing ray. "
        "Try: python3 -m pip install 'ray[default]'"
    )

import pytest
import numpy as np


pytestmark = [pytest.mark.xfail]


def test_all_reduce_circular_communication():
    num_workers = 20
    workers = [Worker(num_workers, i, None) for i in range(num_workers)]

    for i in range(num_workers):
        workers[i].set_friend(workers[(i - 1) % num_workers])

    weight_matrix_shape = (3, 60)
    bias_matrix_shape = (3, 60)

    weight_all_reduced = np.zeros(weight_matrix_shape)
    bias_all_reduced = np.zeros(bias_matrix_shape)
    # setting up gradients for each worker
    for i in range(num_workers):
        workers[i].w_gradients = [(i + 1) * np.ones(weight_matrix_shape)]
        workers[i].b_gradients = [(i + 1) * np.ones(bias_matrix_shape)]
        workers[i].calculate_gradients_partitions()

        # summing the gradients
        weight_all_reduced += workers[i].w_gradients[0]
        bias_all_reduced += workers[i].b_gradients[0]

    weight_all_reduced /= num_workers
    bias_all_reduced /= num_workers

    # first run
    update_id = 0
    for node in range(num_workers - 1):
        if node == num_workers - 2:
            for worker_id in range(num_workers):
                partition_id = (update_id + worker_id - 1) % num_workers
                w_gradients, b_gradients = workers[
                    worker_id
                ].friend.receive_array_partitions(update_id)
                workers[worker_id].friend_weight_gradient_list = w_gradients
                workers[worker_id].friend_bias_gradient_list = b_gradients
                workers[worker_id].update_partitions(
                    partition_id=partition_id, reduce=True, avg_gradients=True
                )
        else:
            for worker_id in range(num_workers):
                partition_id = (update_id + worker_id - 1) % num_workers
                w_gradients, b_gradients = workers[
                    worker_id
                ].friend.receive_array_partitions(update_id)
                workers[worker_id].friend_weight_gradient_list = w_gradients
                workers[worker_id].friend_bias_gradient_list = b_gradients
                workers[worker_id].update_partitions(
                    partition_id=partition_id, reduce=True, avg_gradients=False
                )
        update_id -= 1

    # second run
    update_id = 1
    for node in range(num_workers - 1):
        for worker_id in range(num_workers):
            partition_id = (update_id + worker_id - 1) % num_workers
            w_gradients, b_gradients = workers[
                worker_id
            ].friend.receive_array_partitions(update_id)
            workers[worker_id].friend_weight_gradient_list = w_gradients
            workers[worker_id].friend_bias_gradient_list = b_gradients
            workers[worker_id].update_partitions(
                partition_id=partition_id, reduce=False, avg_gradients=False
            )
        update_id -= 1

    # checking for equality of parameters
    CHECK_PARAMS = True
    for i in range(num_workers):
        CHECK_PARAMS = (
            CHECK_PARAMS and (weight_all_reduced == workers[i].w_gradients).all()
        )
        CHECK_PARAMS = (
            CHECK_PARAMS and (bias_all_reduced == workers[i].b_gradients).all()
        )

    assert CHECK_PARAMS, "Parameters are not same after circular all-reduce."
