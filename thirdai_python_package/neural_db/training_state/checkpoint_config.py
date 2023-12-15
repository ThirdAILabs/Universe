import json
from pathlib import Path

from ..utils import assert_file_exists
from dataclasses import dataclass


class DataSourceCheckpointConfig:
    def __init__(self, checkpoint_dir: Path, type: str):
        self.source_checkpoint_location = checkpoint_dir / f"{type}_source.csv"
        self.source_arguments_location = checkpoint_dir / f"{type}_arguments.json"

    def args(self):
        return {
            "source_checkpoint_location": str(self.source_checkpoint_location),
            "source_arguments_location": str(self.source_arguments_location),
        }

    def assert_data_source_exists(self):
        assert_file_exists(self.source_checkpoint_location)
        assert_file_exists(self.source_arguments_location)


class DirConfig:
    def __init__(self, checkpoint_dir: Path):
        # Checkpoint dir here refers to model specific directory
        self.checkpoint_dir = checkpoint_dir

        self.neuraldb_model_checkpoint_location = self.checkpoint_dir / "model.udt"
        self.intro_datasource_config = DataSourceCheckpointConfig(
            checkpoint_dir=self.checkpoint_dir, type="intro"
        )
        self.train_datasource_config = DataSourceCheckpointConfig(
            checkpoint_dir=self.checkpoint_dir, type="train"
        )

        self.save_arguments_checkpoint_location = (
            self.checkpoint_dir / "save_arguments.json"
        )
        self.intro_args_checkpoint_location = self.checkpoint_dir / "intro_args.json"
        self.train_args_checkpoint_location = self.checkpoint_dir / "training_args.json"
        self.train_state_checkpoint_location = (
            self.checkpoint_dir / "training_state.json"
        )

    def args(self):
        return {
            "neuraldb_model_checkpoint_location": str(
                self.neuraldb_model_checkpoint_location
            ),
            "intro_source_args": self.intro_datasource_config.args(),
            "train_source_args": self.train_datasource_config.args(),
            "save_arguments_checkpoint_location": str(
                self.save_arguments_checkpoint_location
            ),
            "intro_args_checkpoint_location": str(self.intro_args_checkpoint_location),
            "train_args_checkpoint_location": str(self.train_args_checkpoint_location),
            "train_state_checkpoint_location": str(
                self.train_state_checkpoint_location
            ),
        }

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
