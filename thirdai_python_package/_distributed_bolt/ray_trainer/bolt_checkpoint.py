from ray.air.checkpoint import Checkpoint
import os
import tempfile
from thirdai._thirdai import bolt_v2 as bolt

from ray.air.constants import MODEL_KEY


class BoltCheckPoint(Checkpoint):
    """A :py:class:`~ray.air.checkpoint.Checkpoint` with Bolt-specific
    functionality.

    Use ``BoltCheckpoint.from_model`` to create this type of checkpoint.
    """

    @classmethod
    def from_model(
        cls,
        model,
    ):
        with tempfile.TemporaryDirectory() as tmpdirname:
            model.save(os.path.join(tmpdirname, MODEL_KEY))

            checkpoint = cls.from_directory(tmpdirname)
            ckpt_dict = checkpoint.to_dict()

        return cls.from_dict(ckpt_dict)

    def get_model(self):
        with self.as_directory() as checkpoint_path:
            return bolt.nn.Model.load(os.path.join(checkpoint_path, MODEL_KEY))
