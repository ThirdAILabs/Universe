import os
import tempfile

from ray.air.checkpoint import Checkpoint
from ray.air.constants import MODEL_KEY
from thirdai._thirdai import bolt_v2 as bolt


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
        """Create a :py:class:`~ray.air.checkpoint.Checkpoint` that stores a Bolt
        model.

        Args:
            model: The Bolt model to store in the checkpoint.

        Returns:
            An :py:class:`BoltCheckPoint` containing the specified ``Bolt-Model``.

        Examples:
            >>> checkpoint = BoltCheckPoint.from_model(bolt_model)

            >>> model = checkpoint.get_model()
        """
        with tempfile.TemporaryDirectory() as tmpdirname:
            model.save(os.path.join(tmpdirname, MODEL_KEY))

            checkpoint = cls.from_directory(tmpdirname)
            ckpt_dict = checkpoint.to_dict()

        return cls.from_dict(ckpt_dict)

    def get_model(self):
        """Retrieve the Bolt model stored in this checkpoint."""
        with self.as_directory() as checkpoint_path:
            return bolt.nn.Model.load(os.path.join(checkpoint_path, MODEL_KEY))
