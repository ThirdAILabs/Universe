import numpy as np
import pytest

pytestmark = [pytest.mark.distributed]


# This test requires the Ray library, but we don't skip it if Ray isn't
# installed because if someone is running it part of the test may be if the
# Ray install is working at all. Marking it only with
# pytestmark.mark.distributed prevents it from running in our normal unit and
# integration test pipeline where ray isn't a dependency.
def test_all_reduce_circular_communication():
    # Do this import here so pytest collection doesn't fail if ray isn't installed
    from thirdai._distributed_bolt.backend.communication.circular import (
        Circular as Circular,
    )

    num_workers = 16
    circular_communicating_workers = [
        Circular(None, i, None, num_workers) for i in range(num_workers)
    ]

    for i in range(num_workers):
        circular_communicating_workers[i].set_friend(
            circular_communicating_workers[(i - 1) % num_workers]
        )

    flattened_weight_matrix_shape = (100,)

    weights_all_reduced_gt = np.zeros(flattened_weight_matrix_shape, dtype="float32")
    # Set up mock initial gradients for each worker
    for i in range(num_workers):
        circular_communicating_workers[i].gradients = np.random.randint(
            100, size=flattened_weight_matrix_shape
        ).astype("float32")
        weights_all_reduced_gt += circular_communicating_workers[i].gradients
        circular_communicating_workers[i].calculate_gradient_partitions()

    weights_all_reduced_gt /= num_workers

    # This code is copied from the function run_circular_cluster_communication in
    # primary_worker.py and the function process_ring in circular.py
    for update_id, reduce in [(num_workers, True), (num_workers + 1, False)]:
        for node in range(num_workers - 1):
            should_avg_gradients = node == num_workers - 2
            for worker_id in range(num_workers):
                partition_id = (update_id + worker_id - 1) % num_workers
                worker = circular_communicating_workers[worker_id]
                worker.friend_gradients = worker.friend.receive_array_partitions(
                    update_id
                )
                worker.update_partitions(
                    partition_id, reduce=reduce, avg_gradients=should_avg_gradients
                )
            update_id -= 1

    for worker_id, worker in enumerate(circular_communicating_workers):
        assert np.array_equal(
            weights_all_reduced_gt,
            circular_communicating_workers[worker_id].gradients,
        )
