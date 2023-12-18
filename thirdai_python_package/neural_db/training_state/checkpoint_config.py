import json
from dataclasses import dataclass
from pathlib import Path

from ..utils import assert_file_exists


class DataSourceCheckpointConfig:
    """
    source_checkpoint_location : file where the data is stored
    source_arguments_location : file used to initialize the DocumentDataSource object that will load the data.
    """

    def __init__(self, checkpoint_dir: Path, name: str):
        self.source_checkpoint_location = checkpoint_dir / f"{name}_source.csv"
        self.source_arguments_location = checkpoint_dir / f"{name}_arguments.json"

    def assert_data_source_exists(self):
        assert_file_exists(self.source_checkpoint_location)
        assert_file_exists(self.source_arguments_location)


class DirectoryConfig:
    """
    This config class is used to maining the paths of the objects which are created while we are tracking the training progress of NeuralDB. There is a directory config for each model that is present in NeuralDB.
    """

    def __init__(self, checkpoint_dir: Path):
        # Checkpoint dir here refers to model specific directory
        self.checkpoint_dir = checkpoint_dir

        self.neuraldb_model_checkpoint_location = self.checkpoint_dir / "model.udt"
        self.intro_datasource_config = DataSourceCheckpointConfig(
            checkpoint_dir=self.checkpoint_dir, name="intro"
        )
        self.train_datasource_config = DataSourceCheckpointConfig(
            checkpoint_dir=self.checkpoint_dir, name="train"
        )

        self.train_and_intro_state_checkpoint_location = (
            self.checkpoint_dir / "train_state.json"
        )

        self.vlc_path = self.checkpoint_dir / "vlc_config.vlc"

    @property
    def intro_source_arguments_location(self):
        return self.intro_datasource_config.source_arguments_location

    @property
    def train_source_arguments_location(self):
        return self.train_datasource_config.source_arguments_location

    @property
    def intro_source_checkpoint_location(self):
        return self.intro_datasource_config.source_checkpoint_location

    @property
    def train_source_checkpoint_location(self):
        return self.train_datasource_config.source_checkpoint_location

    def assert_checkpoint_source_exists(self):
        self.intro_datasource_config.assert_data_source_exists()
        self.train_datasource_config.assert_data_source_exists()
        assert_file_exists(self.neuraldb_model_checkpoint_location)


@dataclass
class CheckpointConfig:
    checkpoint_dir: Path
    resume_from_checkpoint: bool = False
    checkpoint_interval: int = 1

    def __post_init__(self):
        if isinstance(self.checkpoint_dir, str):
            self.checkpoint_dir = Path(self.checkpoint_dir)
            return
        if isinstance(self.checkpoint_dir, Path):
            return
        else:
            raise TypeError(
                "The 'checkpoint_dir' should be either a 'str' or 'pathlib.Path', but"
                f" received: {type(self.checkpoint_dir)}"
            )
