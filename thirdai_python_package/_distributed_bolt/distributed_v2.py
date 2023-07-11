import numpy as np
from thirdai._thirdai import bolt_v2 as bolt

from .utils import check_torch_installed


class DistributedTrainer(bolt.train.Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Note: We need to disable sparse updates neural network updates as after allreduce
        # during sparse training, we only update the parameters selected by hash tables, rather we
        # need to update all the parameters, since during all-reduce some other neuron could be non-zero
        # too.
        self.model.disable_sparse_parameter_updates()

        import ray

        if not ray.is_initialized():
            raise ValueError(
                "Ray is not initialized. Bolt's distributed training needs acess to a ray cluster!"
            )
        check_torch_installed()

    def train_on_batch(self, inputs, labels, learning_rate):
        # TODO(pratik): Add a check here, so these functions can only be called inside worker-group

        import torch
        import torch.distributed as dist
        from ray.air import session

        num_workers = session.get_world_size()

        self.model.train_on_batch(inputs, labels)

        # This barrier synchronizes different training loops for workers before all-reduce operation.
        dist.barrier()

        gradients = torch.from_numpy(np.array(self.model.get_gradients()))
        dist.all_reduce(gradients)
        self.model.set_gradients(gradients)

        self.model.update_parameters(learning_rate=learning_rate)

    def train(self, train_data, train_labels, learning_rate):
        import torch
        import torch.distributed as dist

        num_batches = len(train_data)

        dist.barrier()
        dist.all_reduce(torch.tensor(num_batches), op=dist.ReduceOp.MIN)

        for batch_id in range(num_batches):
            self.train_on_batch(
                train_data[batch_id], train_labels[batch_id], learning_rate
            )
