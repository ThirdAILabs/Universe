import time
import copy
import numpy as np
import ray
from ray.exceptions import RayActorError, RayError, RayTaskError
from typing import Any, Callable, Dict, Iterator, List, Mapping, Optional, Tuple, Union
class ResultOrError:

    def __init__(self, result: Any = None, error: Exception = None):

        # Note(pratik) : None is a valid result if the remote function
        # does not return anything.
        self._result = result
        # Easier to handle if we show the user the original error.
        self._error = (
            error.as_instanceof_cause() if isinstance(error, RayTaskError) else error
        )

    @property
    def ok(self):
        return self._error is None

    def get(self):
        """Returns the result or the error."""
        if self._error:
            return self._error
        else:
            return self._result

class CallResult:

    worker_id: int
    result_or_error: ResultOrError

    @property
    def ok(self):
        """Passes through the ok property from the result_or_error."""
        return self.result_or_error.ok

    def get(self):
        """Passes through the get method from the result_or_error."""
        return self.result_or_error.get()


class RemoteCallResults:

    class _Iterator:
        """An iterator over the results of a remote call."""

        def __init__(self, call_results: List[CallResult]):
            self._call_results = call_results

        def __iter__(self) -> Iterator[CallResult]:
            return self

        def __next__(self) -> CallResult:
            if not self._call_results:
                raise StopIteration
            return self._call_results.pop(0)

    def __init__(self):
        self.result_or_errors: List[CallResult] = []

    def add_result(self, worker_id: int, result_or_error: ResultOrError):
        self.result_or_errors.append(CallResult(worker_id, result_or_error))

    def __iter__(self) -> Iterator[ResultOrError]:
        # Shallow copy the list.
        return self._Iterator(copy.copy(self.result_or_errors))

    def ignore_errors(self) -> Iterator[ResultOrError]:
        return self._Iterator([r for r in self.result_or_errors if r.ok])

    def ignore_ray_errors(self) -> Iterator[ResultOrError]:
        return self._Iterator(
            [r for r in self.result_or_errors if not isinstance(r.get(), RayActorError)]
        )


class FaultTolerantWorkerManager:
    class _WorkerState:
        # whether this worker is in healthy state
        is_healthy: bool = True

    def __init__(self, workers, max_remote_requests_in_flight_per_worker, init_id):
        # starting worker id from 1, so as to leave worker id 0 to primary worker
        self.next_id = init_id

        self.workers = {}
        self.remote_worker_states = {}
        self.add_workers(workers)

        self.in_flight_req_to_worker_id: Mapping[ray.ObjectRef, int] = {}

        self._max_remote_requests_in_flight_per_worker = (
            max_remote_requests_in_flight_per_worker
        )

        self.num_worker_restarts = 0

    def workers(self):
        return self.workers

    def add_workers(self, workers):
        for worker in workers:
            self.workers[self.next_id] = worker
            self.remote_worker_states[self.next_id] = self._WorkerState
            self.next_id += 1

    def num_workers(self):
        return len(self.workers)

    def num_healthy_workers(self):
        return sum([s.is_healthy for s in self.remote_worker_states.values()])

    def is_worker_healthy(self, worker_id):
        if worker_id not in self.remote_worker_states:
            raise ValueError(f"Unknown worker id: {worker_id}")
        return self.remote_worker_states[worker_id].is_healthy

    def call_workers(
        self,
        func: Union[Callable[[Any], Any], List[Callable[[Any], Any]]],
        *,
        remote_worker_ids: List[int] = None,
    ) -> List[ray.ObjectRef]:
        if isinstance(func, list):
            assert len(remote_worker_ids) == len(
                func
            ), "Funcs must have the same number of callables as worker indices."

        if remote_worker_ids is None:
            remote_worker_ids = list(self.workers.keys())

        if isinstance(func, list):
            calls = [
                self.workers[i].apply.remote(f) for i, f in zip(remote_worker_ids, func)
            ]
        else:
            calls = [self.workers[i].apply.remote(func) for i in remote_worker_ids]

        return calls

    def fetch_result(
        self,
        *,
        remote_worker_ids: List[int],
        remote_calls: List[ray.ObjectRef],
        timeout_seconds: int = None,
        return_obj_refs: bool = False,
    ) -> Tuple[List[ray.ObjectRef], RemoteCallResults]:
        timeout = float(timeout_seconds) if timeout_seconds is not None else None
        ready, _ = ray.wait(
            remote_calls,
            num_returns=len(remote_calls),
            timeout=timeout,
            # Make sure remote results are fetched locally in parallel.
            fetch_local=not return_obj_refs,
        )

        # Remote data should already be fetched to local object store at this point.
        remote_results = RemoteCallResults()
        for r in ready:
            # Find the corresponding worker ID for this remote call.
            worker_id = remote_worker_ids[remote_calls.index(r)]

            # If caller wants ObjectRefs, return directly without resolve them.
            if return_obj_refs:
                remote_results.add_result(worker_id, ResultOrError(result=r))
                continue

            try:
                result = ray.get(r)
                remote_results.add_result(worker_id, ResultOrError(result=result))
            except Exception as e:
                # Return error to the user.
                remote_results.add_result(worker_id, ResultOrError(error=e))

                # Mark the worker as unhealthy.
                # It may very likely be better to use RayActorError here.
                if isinstance(e, RayError):
                    # Take this worker out of service and wait for Ray Core to
                    # restore it.
                    if self.is_worker_healthy(worker_id):
                        print(
                            f"Ray error, taking worker {worker_id} out of service. "
                            f"{str(e)}"
                        )
                    self.set_worker_state(worker_id, healthy=False)
                else:
                    # ActorManager should not handle application level errors.
                    pass

        return ready, remote_results

    def _filter_func_and_remote_worker_id_by_state(
        self,
        func: Union[Callable[[Any], Any], List[Callable[[Any], Any]]],
        remote_worker_ids: List[int],
    ):
        if isinstance(func, list):
            assert len(remote_worker_ids) == len(
                func
            ), "Func must have the same number of callables as remote worker ids."
            # We are given a list of functions to apply.
            # Need to filter the functions together with worker IDs.
            temp_func = []
            temp_remote_worker_ids = []
            for f, i in zip(func, remote_worker_ids):
                if self.is_worker_healthy(i):
                    temp_func.append(f)
                    temp_remote_worker_ids.append(i)
            func = temp_func
            remote_worker_ids = temp_remote_worker_ids
        else:
            # Simply filter the worker IDs.
            remote_worker_ids = [i for i in remote_worker_ids if self.is_worker_healthy(i)]

        return func, remote_worker_ids

    def foreach_worker(
        self,
        func: Union[Callable[[Any], Any], List[Callable[[Any], Any]]],
        *,
        healthy_only=True,
        remote_worker_ids: List[int] = None,
        timeout_seconds=None,
        return_obj_refs: bool = False,
    ) -> RemoteCallResults:
        
        remote_worker_ids = remote_worker_ids or list(self.workers.keys())
        if healthy_only:
            func, remote_worker_ids = self._filter_func_and_remote_worker_id_by_state(
                func, remote_worker_ids
            )

        remote_calls = self.__call_workers(
            func=func,
            remote_worker_ids=remote_worker_ids,
        )

        _, remote_results = self.__fetch_result(
            remote_worker_ids=remote_worker_ids,
            remote_calls=remote_calls,
            timeout_seconds=timeout_seconds,
            return_obj_refs=return_obj_refs,
        )

        return remote_results

    def probe_unhealthy_workers(self) -> List[int]:

        unhealthy_worker_ids = [
            worker_id
            for worker_id in self.workers().keys()
            if not self.is_worker_healthy(worker_id)
        ]

        if not unhealthy_worker_ids:
            return []
        
        remote_results = self.foreach_worker(
            func=lambda worker: worker.ping(),
            remote_worker_ids=unhealthy_worker_ids
            healthy_only=False
        )

        restored = []
        for result in remote_results:
            worker_id = result.worker_id
            if result.ok:
                # Yay, mark this worker as healthy
                restored.append(worker_id)
                self.set_worker_state(worker_id, healthy=True)
                self.num_worker_restarts += 1
            else:
                pass
        
        return restored







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
        all_workers_have_next_batch = True
        try:
            all_workers_have_next_batch = self._compute_and_store_next_batch_gradients()
            self._communicate()
            self._update_parameters()
            self._log_post_batch(epoch)
            self.batch_id_within_epoch += 1
        except RayError:
            print("Some Error happened")
        else:
            pass
        return all_workers_have_next_batch

    def move_to_next_epoch(self):
        self.batch_id_within_epoch = 0
        ray.get([worker.move_to_next_epoch.remote() for worker in self.workers])

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
