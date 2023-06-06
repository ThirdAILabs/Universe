from dataclasses import dataclass

import ray
from ray.train._internal.worker_group import WorkerGroup
from ray.train.backend import Backend, BackendConfig


@dataclass
class BoltBackendConfig(BackendConfig):
    """
    Configuration for Bolt Backend Config

    This class initializes the gloo's collective group with
    world-size, rank and with a group name of "default". Each
    of worker in worker-group connects to this communication
    group as done in `on_start` function.
    """

    @property
    def backend_cls(self):
        return _BoltBackend


def _init_gloo_group(rank, world_size, group_name):
    import ray.util.collective as col
    from ray.util.collective.types import Backend

    col.init_collective_group(
        world_size=world_size,
        rank=rank,
        backend=Backend.GLOO,
        group_name=group_name,
    )


def _destroy_gloo_group(group_name: str = "default"):
    import ray.util.collective as col

    col.destroy_collective_group(group_name)


class _BoltBackend(Backend):
    def on_start(self, worker_group: WorkerGroup, backend_config: BackendConfig):
        # Only gloo backend is available now
        # we wont be supporting linear and circular here
        # initlialize a gloo group
        setup_futures = []
        for i in range(len(worker_group)):
            setup_futures.append(
                worker_group.execute_single_async(
                    i,
                    _init_gloo_group,
                    rank=i,
                    world_size=len(worker_group),
                    group_name="default",
                )
            )
        ray.get(setup_futures)

    def on_shutdown(self, worker_group: WorkerGroup, backend_config: BackendConfig):
        worker_group.execute(_destroy_gloo_group, group_name="default")
