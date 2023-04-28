from thirdai._thirdai import bolt_v2 as bolt
import numpy as np


def _modify_bolt_v2_model():
    def _communicate(self, num_workers):
        import ray.util.collective as col
        from ray.util.collective.types import ReduceOp

        # Assuming we have APIs for model as we had for Distributed Training Wrapper
        # Note(pratik): We havn'e added abarrier here, assuming gloo rendezvous
        # should be able to synchronize all of them.
        gradients = np.array(self.get_gradients())
        col.allreduce(
            tensor=gradients,
            group_name="default",
            ReduceOp=ReduceOp.SUM,
        )
        gradients /= num_workers
        self.set_gradients(gradients)

    def _distribute(self, num_workers):
        import ray.util.collective as col
        from ray.util.collective.types import ReduceOp

        params = np.array(self.params())
        col.allreduce(
            tensor=params,
            group_name="default",
            ReduceOp=ReduceOp.SUM,
        )
        params /= num_workers
        self.set_params(params)

    setattr(bolt.nn.Model, "distribute", _distribute)
    setattr(bolt.nn.Model, "communicate", _communicate)
