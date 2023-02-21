import time
from typing import Any, Callable, List, Optional, Union

import numpy as np
import ray
from ray.exceptions import RayError


class TrainStateManager:
    """
    This class implements a trainer, which controls the trainings,
    expose high level APIs for trainings, predict.
    """

    def __init__(
        self,
        workers,
        primary_worker,
        logging,
        communication_type,
        train_source,
        train_config,
        worker_manager,
    ):
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
        self.worker_manager = worker_manager
        # This tracks the total number of batches completed in this epoch for
        # the distributed job.
        # This differs from the batch count on each worker, which just tracks
        # the current batch within the current dataset on the worker, which will
        # be different if each worker has multiple datasets streamed in, or if
        # something causes a worker to be restarted in the middle of training.
        self.batch_id_within_epoch = 0

        self.train_source = train_source
        self.train_config = train_config
        self.bolt_computation_time = 0
        self.averaging_and_communication_time = 0

    def set_friend_for_primary_worker(self):
        if self.communication_type == "circular":
            self.check_worker_availability_and_call(
                func=lambda worker: worker.set_friend(
                    self.workers[len(self.workers) - 1]
                ),
                # remote_worker_ids=[0] implies this function will
                # only run for worker with id=0/primary worker
                remote_worker_ids=[0],
            )

    def check_worker_availability_and_call(
        self,
        func: Union[Callable[[Any], Any], List[Callable[[Any], Any]]],
        *,
        remote_worker_ids: Optional[List[int]] = None,
        timeout_seconds=None,
        fetch_local: bool = True,
    ):
        self.check_worker_availability()
        return self.worker_manager.foreach_worker(
            func=func,
            remote_worker_ids=remote_worker_ids,
            timeout_seconds=timeout_seconds,
            fetch_local=fetch_local,
        )

    def run_linear_cluster_communication(self):
        """
        This function implements the linear way of communicating between the node.
        In this way of communication, each of the worker calculates their gradients,
        send their gradients to the supervisor and the supervisor sums the gradients,
        averages it and and send the gradients back to the workers.

        :param workers: batch number for the particular worker with worker id (id).
        :type workers: int
        """
        gradients_list = []
        # fetch_local = false, for linear communication to make sure, we don't
        # spill(https://docs.ray.io/en/latest/ray-core/objects/object-spilling.html)
        # gradients during communication, and do communication one by one
        # this doesn't hurt our performance as communication would anyway get queued
        # However, if we spill the ray objects, it would significantly hurt our performance
        for gradients in self.check_worker_availability_and_call(
            lambda worker: worker.get_calculated_gradients(), fetch_local=False
        ).get():
            if gradients.ok:
                gradients_list.append(gradients.get())
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
        self.check_worker_availability_and_call(
            lambda worker: worker.receive_gradients(gradient_averages_ref)
        )
        del gradient_averages_ref

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
                self.check_worker_availability_and_call(
                    lambda worker: worker.process_ring(
                        update_id, avg_gradients=should_avg_gradients, reduce=reduce
                    )
                )
                update_id -= 1

    def get_remote_worker_state(self, worker_with_model):
        # function is called on only one worker, hence just checking for index 0
        remote_bolt_graph_model = (
            self.check_worker_availability_and_call(
                lambda worker: worker.get_model(should_save_optimizer=True),
                remote_worker_ids=[worker_with_model],
            )
            .get()
            .pop()
        )
        remote_train_source_pointers = (
            self.check_worker_availability_and_call(
                lambda worker: worker.get_current_chunk_and_batch(),
                remote_worker_ids=[worker_with_model],
            )
            .get()
            .pop()
        )
        return remote_bolt_graph_model, remote_train_source_pointers

    def prepare_restored_worker_for_training(
        self, bolt_graph_model_ref, chunk_start_index, batch_to_run, restored_workers
    ):
        # At least one of the worker have model
        # ask each worker to get model from that worker
        # by passing in the model ref

        self.check_worker_availability_and_call(
            lambda worker: worker.prepare_for_training(
                bolt_graph=bolt_graph_model_ref,
                chunk_start_index=chunk_start_index,
                batch_to_run=batch_to_run,
            ),
            remote_worker_ids=restored_workers,
        )
        # Primary worker is restored when we are doing circular
        # communication, we need to set the friend because we are not
        # passing it in constructor
        if 0 in restored_workers and self.communication_type == "circular":
            self.check_worker_availability_and_call(
                func=lambda worker: worker.set_friend(
                    self.workers[len(self.workers) - 1]
                ),
                remote_worker_ids=[0],
            )

    def check_worker_availability(self, worker_wait_time_out=1000):
        next_retry_wait_seconds = 1
        while (
            self.worker_manager.num_healthy_workers()
            < self.worker_manager.num_workers()
            and next_retry_wait_seconds < worker_wait_time_out
        ):
            self.logging.info(
                f"Probing unhealthy worker after {next_retry_wait_seconds} seconds!! Total workers:{self.worker_manager.num_workers()}, Unhealthy workers:{self.worker_manager.num_workers()-self.worker_manager.num_healthy_workers()}"
            )

            time.sleep(next_retry_wait_seconds)
            next_retry_wait_seconds *= 2
            # Find workers which are restored
            restored_workers = self.worker_manager.probe_unhealthy_workers()
            self.logging.info(
                f"Preparing restored workers for training: {restored_workers}"
            )
            if len(restored_workers) > 0:
                # Find a worker which have model
                worker_with_model = self.worker_manager.get_worker_with_model()

                (
                    remote_bolt_graph_model,
                    remote_train_source_pointers,
                ) = self.get_remote_worker_state(worker_with_model)

                # Check whether this worker doesn't fails during model fetch
                if remote_train_source_pointers.ok and remote_bolt_graph_model.ok:
                    chunk_start_index, batch_to_run = remote_train_source_pointers.get()
                    bolt_graph_model = remote_bolt_graph_model.get()
                else:
                    self.logging.info("Worker {worker_with_model} failed during fetching model. Need to refetch")
                    continue

                bolt_graph_model_ref = ray.put(bolt_graph_model)
                # We are assuming atleast one of the worker have model
                if worker_with_model != None:
                    self.logging.info("Workers are prepared for trained {restored_workers}")
                    self.prepare_restored_worker_for_training(
                        bolt_graph_model_ref,
                        chunk_start_index,
                        batch_to_run,
                        restored_workers,
                    )
                else:
                    raise NotImplementedError(
                        f"None of the workers are healthy. Distributed BOLT couldn't restart the training. Restart the training again from last saved state."
                    )
        if next_retry_wait_seconds >= worker_wait_time_out:
            raise TimeoutError(
                f"Waited for {worker_wait_time_out}! check_worker_availability timed out!"
            )

    def train_batch(self, epoch):
        """
        Trains the model and returns whether all workers have a next batch.
        """
        all_workers_have_next_batch = self._compute_and_store_next_batch_gradients()
        self._communicate()
        self._update_parameters()
        self._log_post_batch(epoch)
        self.batch_id_within_epoch += 1
        return all_workers_have_next_batch

    def move_to_next_epoch(self):
        self.batch_id_within_epoch = 0
        self.check_worker_availability_and_call(
            lambda worker: worker.move_to_next_epoch()
        )

    def freeze_hash_tables(self):
        self.check_worker_availability_and_call(
            lambda worker: worker.freeze_hash_tables()
        )

    def _compute_and_store_next_batch_gradients(self):
        """
        Calls compute_and_store_batch_gradients function on each of the
        workers and returns whether all workers that respond have a next batch.
        """
        start_calculating_gradients_time = time.time()
        has_next_batches = self.check_worker_availability_and_call(
            lambda worker: worker.compute_and_store_next_batch_gradients()
        ).get()
        self.bolt_computation_time += time.time() - start_calculating_gradients_time
        return all([result.get() for result in has_next_batches if result.ok])

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
            self.check_worker_availability_and_call(
                lambda worker: worker.receive_gradients()
            )
        elif self.communication_type == "gloo":
            self.check_worker_availability_and_call(
                lambda worker: worker.receive_gradients()
            )

        self.averaging_and_communication_time += time.time() - start_communication_time

    def _update_parameters(self):
        """
        Calls each update_parameters on each worker to update parameters
        """
        start_update_parameter_time = time.time()
        self.check_worker_availability_and_call(
            lambda worker: worker.update_parameters()
        )
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
