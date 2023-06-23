import numpy as np
from thirdai._thirdai import bolt_v2 as bolt


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

    def train_on_batch(self, inputs, labels, learning_rate):
        # TODO(pratik): Add a check here, so these functions can only be called inside worker-group

        import ray.util.collective as col
        from ray.air import session
        from ray.util.collective.types import ReduceOp

        if not col.is_group_initialized("default"):
            raise RuntimeError(
                "Gloo group not initialized. Make sure to pass in BoltBackendConfig as backend_config to BoltTrainer"
            )

        num_workers = session.get_world_size()

        self.model.train_on_batch(inputs, labels)

        # Until each of the worker calls this barrier function, we won't be calling all-reduce. We
        # need this since we need all worker to be at gloo's rendezvous before we start
        # communicating, else gloo might timeout waiting for all the workers.
        col.barrier(group_name="default")

        gradients = np.array(self.model.get_gradients())
        col.allreduce(
            tensor=gradients,
            group_name="default",
            op=ReduceOp.SUM,
        )
        gradients /= num_workers
        self.model.set_gradients(gradients)
        self.model.update_parameters(learning_rate=learning_rate)
