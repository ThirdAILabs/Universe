import numpy as np
from thirdai._thirdai import bolt_v2 as bolt


def _modify_bolt_v2_model():
    def _step(self, inputs, labels):
        # TODO(pratik): Add a check here, so these functions can only be called inside worker-group
        import ray.util.collective as col
        from ray.air import session
        from ray.util.collective.types import ReduceOp

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
        self.model.update_parameters(learning_rate=0.001)

    def _distribute(self):
        # TODO(pratik): Add a check here, so these functions can only be called inside worker-group
        import ray.util.collective as col
        from ray.air import session
        from ray.util.collective.types import ReduceOp

        num_workers = session.get_world_size()

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
