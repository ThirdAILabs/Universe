from ray.air.checkpoint import Checkpoint
import os
import bolt


class BoltCheckPoint(Checkpoint):
    """A :py:class:`~ray.air.checkpoint.Checkpoint` with Bolt-specific
    functionality.

    Use ``BoltCheckpoint.from_model`` to create this type of checkpoint.
    """

    @classmethod
    def from_model(
        cls,
        model,
        path,
    ):
        model.save(os.path.join(path, "model.bolt"))

        checkpoint = cls.from_directory(path)

        return checkpoint

    def get_model(self):
        with self.as_directory() as checkpoint_path:
            bolt_model = bolt.nn.Model(os.path.join(checkpoint_path, "model.bolt"))
        return bolt_model
