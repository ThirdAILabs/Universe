import numpy as np
import ray
import time
from typing import Tuple, Any, Optional, Dict, List
from thirdai._distributed_bolt.backend.worker import Worker


@ray.remote(max_restarts=2)
class ReplicaWorker(Worker):
    """This is a ray remote class(Actor). Read about them here.
    (https://docs.ray.io/en/latest/ray-core/actors.html)

    ReplicaWorker is a ray actor which inherits all the function
    of the Worker Class. As the name suggests, it is a replica
    worker and will be reproduced on all the node for parallel
    computations.

    We are using a replica worker class in place of directly using
    worker class, as Primary worker inherits Worker Class and A Ray
    Actor Class can't be inherited.

    Args:
        Worker (Worker): Inherits the worker Class
    """

    def __init__(
        self,
        no_of_workers: int,
        id: int,
        primary_worker,
    ):
        """Calls the constructor for Worker

        Args:
            layers (List[int]): List of layer dimensions.
            config (Dict):  configuration file dictionary
            no_of_workers (int): number of workers in training
            id (int): id of this particular replica worker
        """
        super().__init__(no_of_workers, id, primary_worker)
