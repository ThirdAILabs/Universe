import numpy as np
from thirdai._thirdai import bolt_v2 as bolt


def _modify_bolt_v2_model():
    def _step(self, inputs, labels, learning_rate):
        # TODO(pratik): Add a check here, so these functions can only be called inside worker-group

        import ray.util.collective as col
        from ray.air import session
        from ray.util.collective.types import ReduceOp

        if not col.is_group_initialized():
            raise RuntimeError(
                "Gloo group not initialized. Call trainer.distribute() before calling step"
            )

        num_workers = session.get_world_size()

        self.model.train_on_batch(inputs, labels)
        gradients = np.array(self.model.get_gradients())
        col.allreduce(
            tensor=gradients,
            group_name="default",
            op=ReduceOp.SUM,
        )
        gradients /= num_workers
        self.model.set_gradients(gradients)
        self.model.update_parameters(learning_rate=learning_rate)

    def _distribute(self):
        import ray

        if not ray.is_initialized():
            raise ValueError(
                "Ray is not initialized. Bolt's distributed training needs acess to a ray cluster!"
            )

        import ray.util.collective as col
        from ray.air import session
        from ray.util.collective.types import ReduceOp

        num_workers = session.get_world_size()

        # Note(pratik): We need to disable sparse updates neural network updates as after allreduce
        # during sparse training, we only update the parameters selected by hash tables, rather we
        # need to update all the parameters, since during all-reduce some other neuron could be non-zero
        # too.
        self.model.disable_sparse_parameter_updates()

        # averaging the gradients
        params = np.array(self.model.get_parameters())
        col.allreduce(
            tensor=params,
            group_name="default",
            op=ReduceOp.SUM,
        )
        params /= num_workers
        self.model.set_parameters(params)

    setattr(bolt.train.Trainer, "distribute", _distribute)
    setattr(bolt.train.Trainer, "step", _step)
