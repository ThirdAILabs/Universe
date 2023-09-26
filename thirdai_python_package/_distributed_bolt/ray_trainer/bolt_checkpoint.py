import os

from ray import train
from ray.air.constants import MODEL_KEY
from thirdai._thirdai import bolt

from ..utils import timed


class UDTCheckPoint(train.Checkpoint):
    """A :py:class:`~ray.train.Checkpoint` with UDT-specific
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
        """Create a :py:class:`~ray.train.Checkpoint` that stores a Bolt
        model with/without optimizer states.

        Args:
            model: The UDT model to store in the checkpoint.

        Returns:
            An :py:class:`UDTCheckPoint` containing the specified ``UDT-Model``.

        Examples:
            >>> checkpoint = UDTCheckPoint.from_model(udt_model, with_optimizers=True): saving with optimizer states
            >>> checkpoint = UDTCheckPoint.from_model(udt_model, with_optimizers=False): saving without optimizer states

            >>> model = dist.UDTCheckPoint.get_model(checkpoint)
        """

        save_dir = os.path.join(train.get_context().get_trial_dir(), "tmp_checkpoint")
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, MODEL_KEY)
        model.checkpoint(save_path) if with_optimizers else model.save(save_path)

        checkpoint = cls.from_directory(save_dir)
        checkpoint.set_metadata({"save_dir": save_dir})

        return checkpoint

    @classmethod
    @timed
    def get_model(cls, checkpoint: train.Checkpoint):
        """Retrieve the UDT model stored in this checkpoint."""
        with checkpoint.as_directory() as checkpoint_path:
            return bolt.UniversalDeepTransformer.load(
                os.path.join(checkpoint_path, MODEL_KEY)
            )


class BoltCheckPoint(train.Checkpoint):
    """A :py:class:`~ray.train.Checkpoint` with Bolt-specific
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
        """Create a :py:class:`~ray.train.Checkpoint` that stores a Bolt
        model with/without optimizer states.

        Args:
            model: The Bolt model to store in the checkpoint.

        Returns:
            An :py:class:`BoltCheckPoint` containing the specified ``Bolt-Model``.

        Examples:
            >>> checkpoint = BoltCheckPoint.from_model(bolt_model, with_optimizers=True): saving with optimizer states
            >>> checkpoint = BoltCheckPoint.from_model(bolt_model, with_optimizers=False): saving without optimizer states

            >>> model = dist.BoltCheckPoint.get_model(checkpoint)
        """
        save_dir = os.path.join(train.get_context().get_trial_dir(), "tmp_checkpoint")
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, MODEL_KEY)
        model.checkpoint(save_path) if with_optimizers else model.save(save_path)

        checkpoint = cls.from_directory(save_dir)
        checkpoint.set_metadata({"save_dir": save_dir})

        return checkpoint

    @classmethod
    @timed
    def get_model(cls, checkpoint: train.Checkpoint):
        """Retrieve the Bolt model stored in this checkpoint."""
        with checkpoint.as_directory() as checkpoint_path:
            return bolt.nn.Model.load(os.path.join(checkpoint_path, MODEL_KEY))
