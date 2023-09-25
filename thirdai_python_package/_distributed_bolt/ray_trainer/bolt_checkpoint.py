import os
import tempfile

from ray.air.checkpoint import Checkpoint
from ray.air.constants import MODEL_KEY
from thirdai._thirdai import bolt

from ..utils import timed


class UDTCheckPoint(Checkpoint):
    """A :py:class:`~ray.air.checkpoint.Checkpoint` with UDT-specific
    functionality.

    Use ``UDTCheckPoint.from_model`` to create this type of checkpoint.
    """

    @classmethod
    @timed
    def from_model(
        cls,
        model,
        with_optimizers=True,
    ):
        """Create a :py:class:`~ray.air.checkpoint.Checkpoint` that stores a Bolt
        model with/without optimizer states.

        Args:
            model: The UDT model to store in the checkpoint.

        Returns:
            An :py:class:`UDTCheckPoint` containing the specified ``UDT-Model``.

        Examples:
            >>> checkpoint = UDTCheckPoint.from_model(udt_model, with_optimizers=True): saving with optimizer states
            >>> checkpoint = UDTCheckPoint.from_model(udt_model, with_optimizers=False): saving without optimizer states

            >>> model = checkpoint.get_model()
        """
        with tempfile.TemporaryDirectory() as tmpdirname:
            save_path = os.path.join(tmpdirname, MODEL_KEY)
            model.checkpoint(save_path) if with_optimizers else model.save(save_path)

            checkpoint = cls.from_directory(tmpdirname)
            ckpt_dict = checkpoint.to_dict()

        return cls.from_dict(ckpt_dict)

    @timed
    def get_model(self):
        """Retrieve the Bolt model stored in this checkpoint."""
        with self.as_directory() as checkpoint_path:
            return bolt.UniversalDeepTransformer.load(
                os.path.join(checkpoint_path, MODEL_KEY)
            )


class BoltCheckPoint(Checkpoint):
    """A :py:class:`~ray.air.checkpoint.Checkpoint` with Bolt-specific
    functionality.

    Use ``BoltCheckpoint.from_model`` to create this type of checkpoint.
    """

    @classmethod
    @timed
    def from_model(
        cls,
        model,
        with_optimizers=True,
    ):
        """Create a :py:class:`~ray.air.checkpoint.Checkpoint` that stores a Bolt
        model with/without optimizer states.

        Args:
            model: The Bolt model to store in the checkpoint.

        Returns:
            An :py:class:`BoltCheckPoint` containing the specified ``Bolt-Model``.

        Examples:
            >>> checkpoint = BoltCheckPoint.from_model(bolt_model, with_optimizers=True): saving with optimizer states
            >>> checkpoint = BoltCheckPoint.from_model(bolt_model, with_optimizers=False): saving without optimizer states

            >>> model = checkpoint.get_model()
        """
        with tempfile.TemporaryDirectory() as tmpdirname:
            save_path = os.path.join(tmpdirname, MODEL_KEY)
            model.checkpoint(save_path) if with_optimizers else model.save(save_path)

            checkpoint = cls.from_directory(tmpdirname)
            ckpt_dict = checkpoint.to_dict()

        return cls.from_dict(ckpt_dict)

    @timed
    def get_model(self):
        """Retrieve the Bolt model stored in this checkpoint."""
        with self.as_directory() as checkpoint_path:
            return bolt.nn.Model.load(os.path.join(checkpoint_path, MODEL_KEY))
