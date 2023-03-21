import time

import numpy as np
import ray


class TrainStateManager:
    """
    This class implements a trainer, which controls the trainings,
    expose high level APIs for trainings, predict.
    """

    def __init__(self, workers, primary_worker, logging, communication_type):
        """
        Initializes the TrainStateManager

        :param workers: List of all the workers which includes the primary worker
        :type workers: List[ray.actor]
        :param primary_worker: Primary Actor
        :type primary_worker: ray.actor
        :param logging:  Logs the Training using circular communication pattern
        :type logging: logging
        :param communication_type: Type of communcation which TrainStateManager would be using
        :type communication_type: string
        """

        self.workers = workers
        self.primary_worker = primary_worker
        self.logging = logging
        self.communication_type = communication_type
        self.logging.info(f"Using {communication_type} method for communication")
        # This tracks the total number of batches completed in this epoch for
        # the distributed job.
        # This differs from the batch count on each worker, which just tracks
        # the current batch within the current dataset on the worker, which will
        # be different if each worker has multiple datasets streamed in, or if
        # something causes a worker to be restarted in the middle of training.
        self.batch_id_within_epoch = 0
        self.updates = 0
        if communication_type == "circular":
            for i in range(len(self.workers)):
                ray.get(
                    self.workers[i].set_friend.remote(
                        self.workers[(i - 1) % (len(self.workers))]
                    )
                )
        self.bolt_computation_time = 0
        self.averaging_and_communication_time = 0

    def run_linear_cluster_communication(self):
        """
        This function implements the linear way of communicating between the node.
        In this way of communication, each of the worker calculates their gradients,
        send their gradients to the supervisor and the supervisor sums the gradients,
        averages it and and send the gradients back to the workers.

        :param workers: batch number for the particular worker with worker id (id).
        :type workers: int
        """

        gradients_list = ray.get(
            [worker.get_calculated_gradients.remote() for worker in self.workers]
        )

        # We initialize the sum of gradient variables by setting them equal to the
        # first set of gradients
        self.gradient_averages = np.array(gradients_list[0])

        for worker_id in range(1, len(gradients_list)):
            self.gradient_averages += gradients_list[worker_id]

        self.gradient_averages /= len(self.workers)

        # Here we are putting the references for averaged gradients in the ray plasma store.
        # This allows us to do just a single copy of the gradient array to shared disk, instead
        # of 1 per worker.
        gradient_averages_ref = ray.put(self.gradient_averages)
        ray.get(
            [
                worker.receive_gradients.remote(gradient_averages_ref)
                for worker in self.workers
            ]
        )

    def run_circular_cluster_communication(self):
        """
        This function first call the workers to compute the gradients on their network
        and then implements Baidu's All Ring All Reduce algorithm for communication.
        Read more about that here:
        https://andrew.gibiansky.com/blog/machine-learning/baidu-allreduce/.
        """

        num_workers = len(self.workers)

        # TODO(Pratik): Clean up this function. It is unclear what update_id
        # is, and the input to process_ring has a strange interaction between
        # reduce and should_avg_gradients. Maybe we can make this an enum,
        # something like [DONT_REDUCE, REDUCE, REDUCE_AND_AVERAGE_GRADIENTS].
        for update_id, reduce in [
            (num_workers, True),
            (num_workers + 1, False),
        ]:
            for node in range(num_workers - 1):
                should_avg_gradients = node == num_workers - 2
                ray.get(
                    [
                        worker.process_ring.remote(
                            update_id, avg_gradients=should_avg_gradients, reduce=reduce
                        )
                        for worker in self.workers
                    ]
                )
                update_id -= 1

    def train_batch(self, epoch):
        """
        Trains the model and returns whether all workers have a next batch.
        """
        all_workers_have_next_batch = self._compute_and_store_next_batch_gradients()
        self._communicate()
        self._update_parameters()
        self._log_post_batch(epoch)
        self.batch_id_within_epoch += 1
        self.updates += 1
        return all_workers_have_next_batch

    def move_to_next_epoch(self):
        self.batch_id_within_epoch = 0
        metrics = ray.get(
            [worker.get_updated_metrics.remote() for worker in self.workers]
        )
        ray.get([worker.move_to_next_epoch.remote() for worker in self.workers])
        return metrics

    def freeze_hash_tables(self):
        ray.get([worker.freeze_hash_tables.remote() for worker in self.workers])

    def _compute_and_store_next_batch_gradients(self):
        """
        Calls compute_and_store_batch_gradients function on each of the
        workers and returns whether all workers have a next batch.
        """
        start_calculating_gradients_time = time.time()
        has_next_batches = ray.get(
            [
                worker.compute_and_store_next_batch_gradients.remote()
                for worker in self.workers
            ]
        )
        self.bolt_computation_time += time.time() - start_calculating_gradients_time
        return all(has_next_batches)

    def _communicate(self):
        """
        Calls primary worker to complete the communication
        and then asks all the worker to recieve the updated gradients in their networks
        """
        start_communication_time = time.time()
        if self.communication_type == "linear":
            self.run_linear_cluster_communication()
        elif self.communication_type == "circular":
            self.run_circular_cluster_communication()
            ray.get([worker.receive_gradients.remote() for worker in self.workers])
        elif self.communication_type == "gloo":
            ray.get([worker.receive_gradients.remote() for worker in self.workers])

        self.averaging_and_communication_time += time.time() - start_communication_time

    def _update_parameters(self):
        """
        Calls each update_parameters on each worker to update parameters
        """
        start_update_parameter_time = time.time()
        ray.get([worker.update_parameters.remote() for worker in self.workers])
        self.bolt_computation_time += time.time() - start_update_parameter_time

    def _log_post_batch(self, epoch):
        """
        Logs the training after every batch using the current minimum training
        epoch across workers and the stored self.batch_id_within_epoch in this
        manager, which counts how many total "batches" (iterations of compute
        gradients, communicate, update parameters) we have completed in this
        epoch so far.
        """
        self.logging.info(
            f"Epoch No: {epoch}, Batch Count: {self.batch_id_within_epoch}, Bolt Computation Time: {self.bolt_computation_time}, Averaging and Communcation Time: {self.averaging_and_communication_time}"
        )

    def validate_and_save_if_best(self):
        return ray.get(self.workers[0].validate_and_save_if_best.remote())
