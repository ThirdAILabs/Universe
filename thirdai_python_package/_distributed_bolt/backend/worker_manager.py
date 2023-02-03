import copy
from dataclasses import dataclass
from typing import Any, Callable, Iterator, List, Mapping, Tuple, Union

import numpy as np
import ray
from ray.exceptions import RayActorError, RayError, RayTaskError


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


@dataclass
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
    def __init__(self):
        self.result_or_errors: List[CallResult] = []

    def get_front(self):
        return self.result_or_errors[0]

    def add_result(self, worker_id: int, result_or_error: ResultOrError):
        self.result_or_errors.append(CallResult(worker_id, result_or_error))

    def get(self):
        return self.result_or_errors


class FaultTolerantWorkerManager:
    @dataclass
    class _WorkerState:
        # whether this worker is in healthy state, a worker being healthy implies it is
        # up and could train. But, it doesn't certainly imply it have the model to train.
        is_healthy: bool = True

    def __init__(self, workers, init_id, logging):
        self.next_id = init_id

        self.workers = {}
        self.remote_worker_states = {}
        self._add_workers(workers)

        self.num_worker_restarts = 0
        self.logging = logging

    def _add_workers(self, workers):
        for worker in workers:
            self.workers[self.next_id] = worker
            self.remote_worker_states[self.next_id] = self._WorkerState()
            self.next_id += 1

    def num_workers(self):
        return len(self.workers)

    def _set_worker_state(self, worker_id: int, healthy: bool):
        if worker_id not in self.remote_worker_states:
            raise ValueError(f"Unknown worker id: {worker_id}")
        self.logging.info(f"Worker {worker_id} is being marked as {healthy}")
        self.remote_worker_states[worker_id].is_healthy = healthy

    def num_healthy_workers(self):
        return sum([s.is_healthy for s in self.remote_worker_states.values()])

    def _is_worker_healthy(self, worker_id):
        if worker_id not in self.remote_worker_states:
            raise ValueError(f"Unknown worker id: {worker_id}")
        return self.remote_worker_states[worker_id].is_healthy

    def get_worker_with_model(self):

        remote_worker_ids = list(self.workers.keys())
        remote_have_model = self.foreach_worker(
            func=lambda worker: worker.has_model(), remote_worker_ids=remote_worker_ids
        ).get()

        for remote_result, worker_id in zip(remote_have_model, remote_worker_ids):
            if remote_result.ok:
                if remote_result.get() is True:
                    return worker_id

        return None

    def _call_workers(
        self,
        func: Union[Callable[[Any], Any], List[Callable[[Any], Any]]],
        *,
        remote_worker_ids: List[int] = None,
    ):

        if isinstance(func, list):
            assert len(remote_worker_ids) == len(
                func
            ), "Funcs must have the same number of callables as actor indices."

        if remote_worker_ids is None:
            remote_worker_ids = list(self.worker.keys())

        if isinstance(func, list):
            calls = [
                self.workers[i].apply.remote(f) for i, f in zip(remote_worker_ids, func)
            ]

        else:
            calls = [self.workers[i].apply.remote(func) for i in remote_worker_ids]

        return calls

    def _fetch_result(
        self,
        *,
        remote_worker_ids: List[int],
        remote_calls: List[ray.ObjectRef],
        timeout_seconds: int = None,
        fetch_local: bool = True,
    ):
        timeout = float(timeout_seconds) if timeout_seconds is not None else None
        ready, _ = ray.wait(
            remote_calls,
            num_returns=len(remote_calls),
            timeout=timeout,
            # Make sure remote results are fetched locally in parallel.
            fetch_local=fetch_local,
        )

        # Remote data should already be fetched to local object store at this point.
        remote_results = RemoteCallResults()
        for r in ready:
            # Find the corresponding worker ID for this remote call.
            worker_id = remote_worker_ids[remote_calls.index(r)]

            # If caller wants ObjectRefs, return directly without resolve them.

            try:
                result = ray.get(r)
                remote_results.add_result(worker_id, ResultOrError(result=result))

            except Exception as err:
                # Return error to the user.
                remote_results.add_result(worker_id, ResultOrError(error=err))

                # Mark the worker as unhealthy.
                # It may very likely be better to use RayActorError here.
                if isinstance(err, RayError):
                    # Take this worker out of service and wait for Ray Core to
                    # restore it.
                    if self._is_worker_healthy(worker_id):
                        print(
                            f"Ray error, taking worker {worker_id} out of service. "
                            f"{str(err)}"
                        )
                        self.logging.warning(
                            f"Ray error, taking worker {worker_id} out of service. "
                            f"{str(err)}"
                        )
                    self._set_worker_state(worker_id, healthy=False)
                else:
                    # WorkerManager should not handle application level errors.
                    self.logging.info(f"Got application level Error:{str(err)}")
                    print(f"Unexpected {err=}, {type(err)=}")
                    raise

        return remote_results

    def foreach_worker(
        self,
        func: Union[Callable[[Any], Any], List[Callable[[Any], Any]]],
        *,
        remote_worker_ids: List[int] = None,
        timeout_seconds=None,
        fetch_local: bool = True,
    ):

        remote_worker_ids = remote_worker_ids or list(self.workers.keys())

        remote_calls = self._call_workers(
            func=func,
            remote_worker_ids=remote_worker_ids,
        )

        remote_results = self._fetch_result(
            remote_worker_ids=remote_worker_ids,
            remote_calls=remote_calls,
            timeout_seconds=timeout_seconds,
            fetch_local=fetch_local,
        )

        return remote_results

    def probe_unhealthy_workers(self):

        unhealthy_worker_ids = [
            worker_id
            for worker_id in self.workers.keys()
            if not self._is_worker_healthy(worker_id)
        ]

        self.logging.info(f"Probing unhealthy worker={unhealthy_worker_ids}")
        if not unhealthy_worker_ids:
            return []

        self.logging.info("Calling ping on unhealthy worker")
        remote_results = self.foreach_worker(
            func=lambda worker: worker.ping(),
            remote_worker_ids=unhealthy_worker_ids,
        ).get()

        self.logging.info("Got remote results")
        restored = []
        for result in remote_results:
            worker_id = result.worker_id
            if result.ok:
                restored.append(worker_id)
                self._set_worker_state(worker_id, healthy=True)
                self.num_worker_restarts += 1
            else:
                pass

        self.logging.info(f"Workers restored {restored}")
        return restored
