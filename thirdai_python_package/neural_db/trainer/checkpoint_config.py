from dataclasses import dataclass
from pathlib import Path

from ..utils import convert_str_to_path


@dataclass
class CheckpointConfig:
    """
    Configuration for checkpointing NeuralDB during document insertion.

    Args:
        checkpoint_dir (Path): Directory where models and related metadata will be stored.
        resume_from_checkpoint (bool, optional): If a checkpoint exists, set to True to resume, else False. Defaults to False.
        checkpoint_interval (int, optional): Number of epochs between checkpoints. Defaults to 1.
    """

    checkpoint_dir: Path
    resume_from_checkpoint: bool = False
    checkpoint_interval: int = 1

    def __post_init__(self):
        self.checkpoint_dir = convert_str_to_path(self.checkpoint_dir)

        # After the completion of training, we store the trained neural db at ndb_trained_path
        self._ndb_trained_path = self.checkpoint_dir / "trained.ndb"

        self._pickled_documents_ids_resource_name_path = (
            self.checkpoint_dir / "documents_ids_resource_name.pkl"
        )

    @property
    def ndb_trained_path(self):
        if self._ndb_trained_path is None:
            raise Exception(
                "Invalid Access: The 'ndb_trained_path' property is only"
                " accessible when called by a NeuralDB object within its valid context."
                " Currently, this property is set to None. Ensure that you are"
                " accessing 'ndb_trained_path' from a properly initialized NeuralDB"
                " instance with a valid configuration."
            )
        return self._ndb_trained_path

    @property
    def pickled_documents_ids_resource_name_path(self):
        if self._pickled_documents_ids_resource_name_path is None:
            raise Exception(
                "Invalid Access: The 'pickled_documents_ids_resource_name_path' property is only"
                " accessible when called by a NeuralDB object within its valid context."
                " Currently, this property is set to None. Ensure that you are"
                " accessing 'pickled_documents_ids_resource_name_path' from a properly"
                " initialized NeuralDB instance with a valid configuration."
            )
        return self._pickled_documents_ids_resource_name_path

    def get_mach_config(self):
        """
        This function sets the attributes specific to neural db to None so that we do not make any bad accesses. Ideally, Model object should have no idea about neural db and hence, it should also not be able to access any attributes that disclose any information about neural db
        """
        config = CheckpointConfig(
            checkpoint_dir=self.checkpoint_dir,
            resume_from_checkpoint=self.resume_from_checkpoint,
            checkpoint_interval=self.checkpoint_interval,
        )
        config._ndb_trained_path = None
        config._pickled_documents_ids_resource_name_path = None
        return config


def generate_checkpoint_configs_for_ensembles(
    config: CheckpointConfig, number_ensembles: int, number_models_per_ensemble: int
):
    """
    We maintain a checkpoint config for each Mach model in the Mixture while training. This is designed so that Mach models can maintain their training state independent of their Mixture which is necessary for distributed training.
    """
    if config:
        return [
            [
                CheckpointConfig(
                    config.checkpoint_dir
                    / f"ensemble_{str(ensemble_id)}"
                    / str(model_id),
                    config.resume_from_checkpoint,
                    config.checkpoint_interval,
                ).get_mach_config()
                for model_id in range(number_models_per_ensemble)
            ]
            for ensemble_id in range(number_ensembles)
        ]
    else:
        return [[None] * number_models_per_ensemble] * number_ensembles
