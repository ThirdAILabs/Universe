import os

import pytest
import ray
import thirdai.distributed_bolt as dist
from distributed_utils import setup_ray


from ray.train.torch import TorchConfig
from thirdai import bolt

def test_neural_db_training():
    def training_loop_per_worker(config):
        
    
    scaling_config = setup_ray()
    trainer = dist.BoltTrainer(
        training_loop_per_worker=training_loop_per_worker,
        training_loop_config={"num_epochs": 5},
        
    )