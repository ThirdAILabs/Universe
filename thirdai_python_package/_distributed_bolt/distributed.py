import os

import numpy as np
import thirdai
from thirdai._thirdai import bolt

from .utils import check_torch_installed, timed


class Communication(bolt.train.Communication):
    def __init__(self):
        # For trampoline classes, we need to explicitly call
        # __init__ of the object rather than just using super()
        bolt.train.Communication.__init__(self)
        check_torch_installed()

    @timed
    def synchronize_workers(self):
        import torch.distributed as dist

        dist.barrier()

    @timed
    def communicate(self, model):
        import torch
        import torch.distributed as dist
        from ray import train

        self.synchronize_workers()

        num_workers = train.get_context().get_world_size()
        gradients = torch.from_numpy(np.array(model.get_gradients()))

        dist.all_reduce(gradients)

        gradients = gradients.numpy() / num_workers
        model.set_gradients(gradients)

    @timed
    def min_num_batches(self, num_batches):
        import torch
        import torch.distributed as dist

        dist.barrier()
        all_reduce_num_batches = torch.tensor(num_batches)
        dist.all_reduce(all_reduce_num_batches, op=dist.ReduceOp.MIN)
        return all_reduce_num_batches

    @timed
    def broadcast_metrics(self, train_metrics):
        import torch.distributed as dist

        dist.barrier()
        dist.broadcast_object_list(train_metrics, src=0)

        return train_metrics


# Note: We need to disable sparse updates neural network updates as after allreduce
# during sparse training, we only update the parameters selected by hash tables, rather we
# need to update all the parameters, since during all-reduce some other neuron could be non-zero
# too.
def adds_distributed_to_bolt():
    def train_distributed(self, *args, **kwargs):
        self.model.disable_sparse_parameter_updates()

        kwargs["comm"] = Communication()
        metrics = self.train(*args, **kwargs)

        self.model.enable_sparse_parameter_updates()

        return metrics

    bolt.train.Trainer.train_distributed = train_distributed

    def udt_train_distributed(self, *args, **kwargs):
        self._get_model().disable_sparse_parameter_updates()

        kwargs["comm"] = Communication()
        metrics = self.train(*args, **kwargs)

        self._get_model().enable_sparse_parameter_updates()

        return metrics

    bolt.UniversalDeepTransformer.train_distributed = udt_train_distributed

    def udt_coldstart_distributed(self, *args, **kwargs):
        self._get_model().disable_sparse_parameter_updates()

        kwargs["comm"] = Communication()
        metrics = self.cold_start(*args, **kwargs)

        self._get_model().enable_sparse_parameter_updates()

        return metrics

    def udt_coldstart_distributed_on_data_source(self, *args, **kwargs):
        self._get_model().disable_sparse_parameter_updates()

        kwargs["comm"] = Communication()
        metrics = self.cold_start_on_data_source(*args, **kwargs)

        self._get_model().enable_sparse_parameter_updates()

        return metrics

    bolt.UniversalDeepTransformer.coldstart_distributed = udt_coldstart_distributed

    bolt.UniversalDeepTransformer.coldstart_distributed_on_data_source = (
        udt_coldstart_distributed_on_data_source
    )

    def generative_model_train_distributed(self, *args, **kwargs):
        self.model.disable_sparse_parameter_updates()

        kwargs["comm"] = Communication()
        metrics = self.train(*args, **kwargs)

        self.model.enable_sparse_parameter_updates()

        return metrics

    bolt.GenerativeModel.train_distributed = generative_model_train_distributed
