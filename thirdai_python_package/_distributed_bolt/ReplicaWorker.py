import numpy as np
import ray
import time
from typing import Tuple, Any, Optional, Dict, List
from .utils import initLogging
from .Worker import Worker


@ray.remote(max_restarts=2)
class ReplicaWorker(Worker):
    """
    This is a ray remote class(Actor). Read about them here.
    (https://docs.ray.io/en/latest/ray-core/actors.html)

    ReplicaWorker is a ray actor which inherits all the function
    of the Worker Class. As the name suggests, it is a replica
    worker and will be reproduced on all the node for parallel
    computations.



    Args:
        layers: List of layer dimensions.
        config: Configuration file for the training
        no_of_workers: Total number of workers
        id: Id for this worker
    """

    def __init__(self, layers: List, config, no_of_workers, id):
        super().__init__(layers, config, no_of_workers, id)
