from __future__ import annotations

import shutil
from pathlib import Path
from typing import Optional, Union

from ..documents import DocumentDataSource
from ..supervised_datasource import SupDataSource
from ..utils import assert_file_exists, move_between_directories, unpickle_from
from .training_progress_tracker import NeuralDbProgressTracker


class SupervisedDataManager:
    def __init__(self, checkpoint_dir: Optional[Path], train_source: SupDataSource):
        self.checkpoint_dir = checkpoint_dir
        self.train_source = train_source

        if self.checkpoint_dir:
            self.train_source_folder = self.checkpoint_dir / "sup_source"
            self.train_source_folder.mkdir(exist_ok=True, parents=True)

    def save(self):
        if self.checkpoint_dir:
            self.train_source.save(self.train_source_folder)
        else:
            raise Exception(
                "Invalid method call: 'save' operation for SupervisedDataManager cannot"
                " be executed because 'checkpoint_dir' is None. Please provide a valid"
                " directory path for 'checkpoint_dir' to proceed with the save"
                " operation."
            )

    @staticmethod
    def load(checkpoint_dir: Path, train_source: Optional[SupDataSource] = None):
        manager = SupervisedDataManager(checkpoint_dir, None)
        manager.train_source = (
            train_source
            if train_source
            else SupDataSource.load(path=manager.intro_source_folder)
        )
        return manager


class InsertDataManager:
    def __init__(
        self,
        checkpoint_dir: Optional[Path],
        intro_source: DocumentDataSource,
        train_source: DocumentDataSource,
    ):
        self.checkpoint_dir = checkpoint_dir
        if self.checkpoint_dir:
            self.intro_source_folder = self.checkpoint_dir / "intro_source"
            self.train_source_folder = self.checkpoint_dir / "train_source"
            self.intro_source_folder.mkdir(exist_ok=True, parents=True)
            self.train_source_folder.mkdir(exist_ok=True)

        self.intro_source = intro_source
        self.train_source = train_source

    def save(self):
        if self.checkpoint_dir:
            self.intro_source.save(path=self.intro_source_folder)
            self.train_source.save(path=self.train_source_folder)
        else:
            raise Exception(
                "Invalid method call: 'save' operation for InsertDataManager cannot"
                " be executed because 'checkpoint_dir' is None. Please provide a valid"
                " directory path for 'checkpoint_dir' to proceed with the save"
                " operation."
            )

    @staticmethod
    def load(
        checkpoint_dir: Path,
        intro_shard: Optional[DocumentDataSource] = None,
        train_shard: Optional[DocumentDataSource] = None,
    ):
        manager = InsertDataManager(checkpoint_dir, None, None)
        manager.intro_source = (
            intro_shard
            if intro_shard
            else DocumentDataSource.load(path=manager.intro_source_folder)
        )
        manager.train_source = (
            train_shard
            if train_shard
            else DocumentDataSource.load(path=manager.train_source_folder)
        )
        return manager


class TrainingDataManager:
    """
    This manager class maintains the data needed by the training progress manager. Supports both saving and loading the data. When the manager is initialized with a checkpoint_dir as None, all save and load throw an error.
    """

    def __init__(
        self,
        checkpoint_dir: Optional[Path],
        model,
        datasource_manager: Union[InsertDataManager, SupervisedDataManager],
        tracker: NeuralDbProgressTracker,
    ):
        # Checkpoint dir here refers to model specific directory
        self.checkpoint_dir = checkpoint_dir

        if self.checkpoint_dir:
            self.model_location = self.checkpoint_dir / "model.pkl"
            self.tracker_folder = self.checkpoint_dir / "tracker"
            self.tracker_folder.mkdir(exist_ok=True, parents=True)

        self.model = model
        self.datasource_manager = datasource_manager
        self.tracker = tracker

    def save(self, save_datasource=True):
        if self.checkpoint_dir:
            self.model.save(path=self.model_location)
            self.tracker.save(path=self.tracker_folder)
            if save_datasource:
                self.datasource_manager.save()

        else:
            raise Exception(
                "Invalid method call: 'save' operation for TrainingDataManager cannot"
                " be executed because 'checkpoint_dir' is None. Please provide a valid"
                " directory path for 'checkpoint_dir' to proceed with the save"
                " operation."
            )

    @property
    def intro_source(self):
        if isinstance(self.datasource_manager, InsertDataManager):
            return self.datasource_manager.intro_source
        else:
            raise Exception(
                "Invalid method call: 'intro_source' operation for TrainingDataManager cannot"
                " be executed because 'datasource_manager' is of the type SupervisedDataManager"
            )

    @property
    def train_source(self):
        return self.datasource_manager.train_source

    def save_without_sources(self):
        if self.checkpoint_dir:
            self.model.save(path=self.model_location)
            self.tracker.save(path=self.tracker_folder)
        else:
            raise Exception(
                "Invalid method call: 'save_without_sources' operation for"
                " TrainingDataManager cannot be executed because 'checkpoint_dir' is"
                " None. Please provide a valid directory path for 'checkpoint_dir' to"
                " proceed with the save operation."
            )

    @staticmethod
    def load(
        checkpoint_dir: Path,
        data_manager: Union[SupervisedDataManager, InsertDataManager],
    ):
        manager = TrainingDataManager(checkpoint_dir, None, None, None)

        try:
            manager.model = unpickle_from(manager.model_location)
        except:
            raise Exception(
                "Could not find a valid Mach model at the path:"
                f" {manager.model_location}"
            )

        manager.datasource_manager = data_manager
        manager.tracker = NeuralDbProgressTracker.load(path=manager.tracker_folder)

        return manager

    def delete_checkpoint(self):
        shutil.rmtree(path=self.checkpoint_dir, ignore_errors=True)

    @staticmethod
    def update_model_and_tracker_from_backup(
        backup_config: TrainingDataManager,
        target_config: TrainingDataManager,
    ):
        assert_file_exists(path=backup_config.model_location)
        assert_file_exists(path=backup_config.tracker_folder)

        shutil.move(
            backup_config.model_location,
            target_config.model_location,
        )

        move_between_directories(
            backup_config.tracker_folder, target_config.tracker_folder
        )

    def copy_with_new_dir(self, new_directory):
        return TrainingDataManager(
            checkpoint_dir=new_directory,
            model=self.model,
            datasource_manager=self.datasource_manager,
            tracker=self.tracker,
        )
