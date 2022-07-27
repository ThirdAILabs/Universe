import numpy as np
import ray
import time
from typing import Tuple, Any, Optional, Dict, List
from .utils import initLogging
from .Worker import Worker



@ray.remote(num_cpus=20, max_restarts=2)
class ReplicaWorker(Worker):
    """
        This is a ray remote class(Actor). Read about them here. 
        (https://docs.ray.io/en/latest/ray-core/actors.html)

        Supervisor is a ray actor which implements higher level 
        abstraction on worker nodes. It controls training on 
        each of the node(which batch number to train) and communication
        between the worker nodes.
        
        
        Args:
            layers: List of layer dimensions.
            workers: List of workers(including the worker on head node) running on different nodes.
    """
    def __init__(
        self, 
        layers: List, 
        config, 
        no_of_workers,
        id

    ):
        super().__init__(layers, config, no_of_workers, id)
    
    


