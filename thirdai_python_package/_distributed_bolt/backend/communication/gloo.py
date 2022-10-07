import ray

import warnings
import sys, os

import ray.util.collective as col
from ray.util.collective.types import Backend, ReduceOp


import numpy as np

from ...utils import set_gradients, get_gradients

class Gloo:
    def __init__(self, model, id, num_workers, group_name):
        self.model = model
        self.id = id
        self.num_workers = num_workers
        self.gradients = []

        # Gloo initialization
        self.group_name = group_name
        col.init_collective_group(num_workers, id, Backend.GLOO, self.group_name)



    def compute_and_store_batch_gradients(self, batch_no):
        self.model.compute_and_store_batch_gradients(batch_no)
        self.gradients = np.array(get_gradients(self.model))


    def receive_gradients(self):
        for gradient_id in range(len(self.gradients)):
            col.allreduce(self.gradients[gradient_id], self.group_name, ReduceOp.SUM)
            self.gradients[gradient_id] /= self.num_workers

        
        set_gradients(self.model, self.gradients)