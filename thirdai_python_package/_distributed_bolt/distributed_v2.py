import numpy as np
from thirdai._thirdai import bolt_v2 as bolt


def _modify_bolt_v2_model():
    def _step(self, inputs, labels, num_workers):
        import ray.util.collective as col
        from ray.util.collective.types import ReduceOp

        # Assuming we have APIs for model as we had for Distributed Training Wrapper
        # Note(pratik): We havn'e added abarrier here, assuming gloo rendezvous
        # should be able to synchronize all of them.
        self.model.train_on_batch(inputs, labels)
        gradients = np.array(self.model.get_values(0))
        col.allreduce(
            tensor=gradients,
            group_name="default",
            op=ReduceOp.SUM,
        )
        gradients /= num_workers
        self.model.set_values(gradients, 0)
        self.model.update_parameters(learning_rate=0.001)

    def _distribute(self, num_workers):
        import ray.util.collective as col
        from ray.util.collective.types import ReduceOp

        params = np.array(self.model.get_values(1))
        col.allreduce(
            tensor=params,
            group_name="default",
            op=ReduceOp.SUM,
        )
        params /= num_workers
        self.model.set_values(params, 1)

    setattr(bolt.train.Trainer, "distribute", _distribute)
    setattr(bolt.train.Trainer, "step", _step)
