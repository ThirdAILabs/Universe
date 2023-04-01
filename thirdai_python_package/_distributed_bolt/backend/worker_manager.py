import copy
from dataclasses import dataclass
from typing import Any, Callable, List, Optional, Union

import ray
from ray.exceptions import RayTaskError


class ResultOrError:
    def __init__(self, result: Any = None, error: Exception = None):
        # None is a valid result if the remote function
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

        self.logging = logging

    def num_workers(self):
        return len(self.workers)


    def foreach_worker(
        self,
        func: Union[Callable[[Any], Any], List[Callable[[Any], Any]]],
        *,
        remote_worker_ids: Optional[List[int]] = None,
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

    def _add_workers(self, workers):
        for worker in workers:
            self.workers[self.next_id] = worker
            self.remote_worker_states[self.next_id] = self._WorkerState()
            self.next_id += 1


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
        timeout_seconds: float = None,
        fetch_local: bool = True,
    ):
        """
        This function receives the calls and worker_ids. It passes the calls to
        ray.wait(https://docs.ray.io/en/latest/ray-core/package-ref.html#ray-wait)
        which in turns returns a list of object which are ready in object store.
        It fetches the object, handles any failure in fetching and returns a RemoteCallResult
        list.

        Args:
            remote_worker_ids (List[int]): list of worker_ids to call remote_calls
            remote_calls (List[ray.ObjectRef]): list of remote_calls for workers
            timeout_seconds (float, optional): time to wait for object to process. Defaults to None.
            fetch_local (bool, optional): make sure remote results are fetched locally in parallel.
                Defaults to True.

        Returns:
            RemoteCallResults: Return from remote function calls
        """
        timeout = timeout_seconds if timeout_seconds is not None else None
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
                #TODO(pratik): Add error handling code when worker faults. 
                
                # Return error to the user.
                remote_results.add_result(worker_id, ResultOrError(error=err))
                

        return remote_results