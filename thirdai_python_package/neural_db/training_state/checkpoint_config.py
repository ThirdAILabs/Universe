from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Union

from ..documents import DocumentDataSource
from ..utils import assert_file_exists, pickle_to, unpickle_from
from .training_progress_tracker import NeuralDbProgressTracker


class DataSourceCheckpointManager:
    """
    source_checkpoint_location : file where the data is stored
    source_arguments_location : file used to initialize the DocumentDataSource object that will load the data.
    """

    def __init__(self, checkpoint_dir: Union[Path, None], source: DocumentDataSource):
        self.checkpoint_dir = checkpoint_dir
        if self.checkpoint_dir:
            self.source_checkpoint_location = self.checkpoint_dir / f"source.csv"
            self.source_arguments_location = self.checkpoint_dir / f"arguments.json"
            self.checkpoint_dir.mkdir(exist_ok=True, parents=True)
        self.source = source

    def assert_data_source_exists(self):
        if self.checkpoint_dir:
            assert_file_exists(self.source_checkpoint_location)
            assert_file_exists(self.source_arguments_location)
        else:
            raise ValueError(
                "Datasource Checkpoint Manager initialized with checkpoint_dir as None."
            )

    def save(self):
        if self.checkpoint_dir:
            self.source.save_to_csv(csv_path=self.source_checkpoint_location)
            with open(self.source_arguments_location, "w") as f:
                json.dump(self.source.initialization_args(), f, indent=4)

    @staticmethod
    def load(checkpoint_dir: Path):
        config = DataSourceCheckpointManager(checkpoint_dir=checkpoint_dir, source=None)
        config.assert_data_source_exists()

        with open(config.source_arguments_location, "r") as f:
            args = json.load(f)
        config.source = DocumentDataSource.load_from_csv(
            csv_path=config.source_checkpoint_location,
            id_column=args["id_column"],
            strong_column=args["strong_column"],
            weak_column=args["weak_column"],
        )
        return config


class ModelCheckpointManager:
    def __init__(self, checkpoint_dir: Union[Path, None], neural_db_model):
        self.checkpoint_dir = checkpoint_dir
        self.model = neural_db_model

    def assert_model_exists(self):
        if self.checkpoint_dir:
            assert_file_exists(self.checkpoint_dir)
        else:
            raise ValueError(
                "Model Checkpoint Manager initialized with checkpoint_dir as None."
            )

    def save(self):
        if self.checkpoint_dir:
            pickle_to(obj=self.model, filepath=self.checkpoint_dir)

    @staticmethod
    def load(path: Path):
        try:
            model = unpickle_from(filepath=path)
            return ModelCheckpointManager(path, model)
        except:
            raise Exception(f"Could not find a valid model at the path{path}")


class TrackerCheckpointManager:
    def __init__(
        self, checkpoint_dir: Union[Path, None], tracker: NeuralDbProgressTracker
    ):
        self.checkpoint_dir = checkpoint_dir
        if checkpoint_dir:
            self.arguments_save_location = checkpoint_dir / "tracker.json"
            self.vlc_save_location = checkpoint_dir / "vlc_config"
        self.tracker = tracker

    def assert_tracker_exists(self):
        if self.checkpoint_dir:
            assert_file_exists(self.arguments_save_location)
            assert_file_exists(self.vlc_save_location)
        else:
            raise ValueError(
                "Tracker Checkpoint Manager initialized with checkpoint_dir as None."
            )

    def save(self):
        if self.checkpoint_dir:
            with open(self.arguments_save_location, "w") as f:
                json.dump(self.tracker.__dict__(), f, indent=4)
                pickle_to(self.tracker.vlc_config, filepath=self.vlc_save_location)

    @staticmethod
    def load(checkpoint_dir: Path):
        config = TrackerCheckpointManager(checkpoint_dir=checkpoint_dir, tracker=None)
        config.assert_tracker_exists()

        with open(config.arguments_save_location) as f:
            args = json.load(f)

        vlc_config = unpickle_from(filepath=config.vlc_save_location)
        config.tracker = NeuralDbProgressTracker.load(
            arguments_json=args, vlc_config=vlc_config
        )
        return config


class TrainingDataCheckpointManager:
    """
    This manager class maintains the data needed by the training progress manager. Supports both saving and loading the data. When the manager is initialized with a checkpoint_dir as None, all save and load operations become a no_op.
    """

    def __init__(
        self,
        checkpoint_dir: Union[Path, None],
        model,
        intro_source: DocumentDataSource,
        train_source: DocumentDataSource,
        tracker: NeuralDbProgressTracker,
    ):
        # Checkpoint dir here refers to model specific directory
        self.checkpoint_dir = checkpoint_dir

        if self.checkpoint_dir:
            self.checkpoint_dir.mkdir(exist_ok=True, parents=True)

        self.model_config = ModelCheckpointManager(
            checkpoint_dir=(
                self.checkpoint_dir / "model.udt" if self.checkpoint_dir else None
            ),
            neural_db_model=model,
        )

        self.intro_config = DataSourceCheckpointManager(
            checkpoint_dir=(
                self.checkpoint_dir / "intro" if self.checkpoint_dir else None
            ),
            source=intro_source,
        )
        self.train_config = DataSourceCheckpointManager(
            checkpoint_dir=(
                self.checkpoint_dir / "train" if self.checkpoint_dir else None
            ),
            source=train_source,
        )

        self.tracker_config = TrackerCheckpointManager(
            checkpoint_dir=self.checkpoint_dir if self.checkpoint_dir else None,
            tracker=tracker,
        )

    def assert_checkpoint_data_exists(self):
        self.model_config.assert_model_exists()
        self.intro_config.assert_data_source_exists()
        self.train_config.assert_data_source_exists()
        self.tracker_config.assert_tracker_exists()

    def save(self):
        self.intro_config.save()
        self.train_config.save()
        self.model_config.save()
        self.tracker_config.save()

    def save_without_sources(self):
        self.model_config.save()
        self.tracker_config.save()

    @staticmethod
    def load(checkpoint_dir: Path):
        manager = TrainingDataCheckpointManager(checkpoint_dir, None, None, None, None)  # type: ignore

        manager.assert_checkpoint_data_exists()

        manager.model_config = ModelCheckpointManager.load(
            path=manager.model_config.checkpoint_dir  # type: ignore
        )
        manager.intro_config = DataSourceCheckpointManager.load(
            checkpoint_dir=manager.intro_config.checkpoint_dir  # type: ignore
        )
        manager.train_config = DataSourceCheckpointManager.load(
            checkpoint_dir=manager.train_config.checkpoint_dir  # type: ignore
        )
        manager.tracker_config = TrackerCheckpointManager.load(
            checkpoint_dir=checkpoint_dir
        )
        return manager

    @staticmethod
    def update_model_and_tracker_from_backup(
        backup_config: TrainingDataCheckpointManager,
        target_config: TrainingDataCheckpointManager,
    ):
        backup_config.model_config.assert_model_exists()
        backup_config.tracker_config.assert_tracker_exists()

        shutil.move(
            backup_config.tracker_config.arguments_save_location,
            target_config.tracker_config.arguments_save_location,
        )

        shutil.move(
            backup_config.model_config.checkpoint_dir,
            target_config.model_config.checkpoint_dir,
        )

    @property
    def intro_source(self):
        return self.intro_config.source

    @property
    def train_source(self):
        return self.train_config.source

    @property
    def model(self):
        return self.model_config.model

    @model.setter
    def model(self, new_model):
        self.model_config.model = new_model

    @property
    def tracker(self):
        return self.tracker_config.tracker

    def copy_with_new_dir(self, new_directory):
        return TrainingDataCheckpointManager(
            checkpoint_dir=new_directory,
            model=self.model,
            intro_source=self.intro_source,
            train_source=self.train_source,
            tracker=self.tracker,
        )


@dataclass
class NDBCheckpointConfig:
    checkpoint_dir: Path
    resume_from_checkpoint: bool = False
    checkpoint_interval: int = 1

    def __post_init__(self):
        if isinstance(self.checkpoint_dir, str):
            self.checkpoint_dir = Path(self.checkpoint_dir)
        elif isinstance(self.checkpoint_dir, Path):
            pass
        else:
            raise TypeError(
                "The 'checkpoint_dir' should be either a 'str' or 'pathlib.Path', but"
                f" received: {type(self.checkpoint_dir)}"
            )

        # Before insert process is started, we first make a checkpoint of the neural db at ndb_checkpoint_path
        self.ndb_checkpoint_path = self.checkpoint_dir / "checkpoint.ndb"
        # After the completion of training, we store the trained neural db at ndb_trained_path
        self.ndb_trained_path = self.checkpoint_dir / "trained.ndb"

        self.pickled_ids_resource_name_path = (
            self.checkpoint_dir / "ids_resource_name.pkl"
        )


@dataclass
class MachCheckpointConfig:
    # TODO(Shubh): NDB and Mach checkpoint config are almost identical. Should we just have one of them?
    checkpoint_dir: Path
    resume_from_checkpoint: bool = False
    checkpoint_interval: int = 1  # no of epochs between saving the checkpoints

    def __post_init__(self):
        if isinstance(self.checkpoint_dir, str):
            self.checkpoint_dir = Path(self.checkpoint_dir)
        elif isinstance(self.checkpoint_dir, Path):
            pass
        else:
            raise TypeError(
                "The 'checkpoint_dir' should be either a 'str' or 'pathlib.Path', but"
                f" received: {type(self.checkpoint_dir)}"
            )


def convert_ndb_checkpoint_config_to_mach(ndb_config: NDBCheckpointConfig):
    if ndb_config is None:
        return None

    return MachCheckpointConfig(
        checkpoint_dir=ndb_config.checkpoint_dir,
        resume_from_checkpoint=ndb_config.resume_from_checkpoint,
        checkpoint_interval=ndb_config.checkpoint_interval,
    )
