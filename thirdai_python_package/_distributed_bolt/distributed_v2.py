import numpy as np
from thirdai._thirdai import bolt as old_bolt
from thirdai._thirdai import bolt_v2 as bolt

from .utils import check_torch_installed


class Communication(bolt.train.Communication):
    def __init__(self):
        bolt.train.Communication.__init__(self)
        check_torch_installed()

    def communicate(self, model):
        import torch
        import torch.distributed as dist
        from ray.air import session

        num_workers = session.get_world_size()
        dist.barrier()

        gradients = torch.from_numpy(np.array(model.get_gradients()))

        dist.all_reduce(gradients)
        gradients = gradients.numpy() / num_workers
        model.set_gradients(gradients)

    def min_num_batches(self, num_batches):
        import torch
        import torch.distributed as dist

        dist.barrier()
        all_reduce_num_batches = torch.tensor(num_batches)
        dist.all_reduce(all_reduce_num_batches, op=dist.ReduceOp.MIN)
        return all_reduce_num_batches


class DistributedTrainer(bolt.train.Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Note: We need to disable sparse updates neural network updates as after allreduce
        # during sparse training, we only update the parameters selected by hash tables, rather we
        # need to update all the parameters, since during all-reduce some other neuron could be non-zero
        # too.
        self.model.disable_sparse_parameter_updates()

    def train_distributed(self, *args, **kwargs):
        kwargs["comm"] = Communication()
        self.train(*args, **kwargs)


def adds_distributed_v2_to_udt():
    def coldstart_distributed_v2(self, *args, **kwargs):
        self._get_model().disable_sparse_parameter_updates()
        kwargs["comm"] = Communication()
        return self.cold_start(*args, **kwargs)

        # TODO(pratik/mritunjay): Enable sparse parameter updates after training.

    old_bolt.UniversalDeepTransformer.coldstart_distributed_v2 = (
        coldstart_distributed_v2
    )

    def train_distributed_v2(self, *args, **kwargs):
        self._get_model().disable_sparse_parameter_updates()
        kwargs["comm"] = Communication()
        return self.train(*args, **kwargs)

        # TODO(pratik/mritunjay): Enable sparse parameter updates after training.

    old_bolt.UniversalDeepTransformer.train_distributed_v2 = train_distributed_v2
