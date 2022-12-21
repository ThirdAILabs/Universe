import os
import sys
import warnings

import numpy as np
import ray
import ray.util.collective as col
from ray.util.collective.types import Backend, ReduceOp


# TODO(pratik): Add tests for gloo, as soon as next version of pygloo is released.
class Gloo:
    def __init__(self, model, id, num_workers, group_name):
        self.model = model
        self.id = id
        self.num_workers = num_workers
        self.gradients = []

        # Gloo initialization
        self.group_name = group_name
        col.init_collective_group(
            world_size=num_workers,
            rank=id,
            backend=Backend.GLOO,
            group_name=self.group_name,
        )

    def compute_and_store_batch_gradients(self, batch_no):
        self.model.compute_and_store_batch_gradients(batch_no)
        self.gradients = np.array(self.model.gradient_reference().get_gradients())

    def receive_gradients(self):
        col.allreduce(
            tensor=self.gradients,
            group_name=self.group_name,
            op=ReduceOp.SUM,
        )
        self.gradients /= self.num_workers

        self.model.gradient_reference().set_gradients(self.gradients)
