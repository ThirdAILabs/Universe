from __future__ import annotations

import shutil
from pathlib import Path
from typing import Optional

from ..documents import DocumentDataSource
from ..utils import assert_file_exists, move_between_directories, unpickle_from
from .training_progress_tracker import NeuralDbProgressTracker


class TrainingDataManager:
    """
    This manager class maintains the data needed by the training progress manager. Supports both saving and loading the data. When the manager is initialized with a checkpoint_dir as None, all save and load throw an error.
    """

    def __init__(
        self,
        checkpoint_dir: Optional[Path],
        model,
        intro_source: DocumentDataSource,
        train_source: DocumentDataSource,
        tracker: NeuralDbProgressTracker,
    ):
        # Checkpoint dir here refers to model specific directory
        self.checkpoint_dir = checkpoint_dir

        if self.checkpoint_dir:
            self.model_location = self.checkpoint_dir / "model.pkl"
            self.intro_source_folder = self.checkpoint_dir / "intro_source"
            self.train_source_folder = self.checkpoint_dir / "train_source"
            self.tracker_folder = self.checkpoint_dir / "tracker"

            self.intro_source_folder.mkdir(exist_ok=True, parents=True)
            self.train_source_folder.mkdir(exist_ok=True)
            self.tracker_folder.mkdir(exist_ok=True)

        self.model = model
        self.intro_source = intro_source
        self.train_source = train_source
        self.tracker = tracker

    def save(self):
        if self.checkpoint_dir:
            self.model.save(path=self.model_location)
            self.intro_source.save(path=self.intro_source_folder)
            self.train_source.save(path=self.train_source_folder)
            self.tracker.save(path=self.tracker_folder)
        else:
            raise Exception(
                "Invalid method call: 'save' operation for TrainingDataManager cannot"
                " be executed because 'checkpoint_dir' is None. Please provide a valid"
                " directory path for 'checkpoint_dir' to proceed with the save"
                " operation."
            )

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
    def load(checkpoint_dir: Path):
        manager = TrainingDataManager(checkpoint_dir, None, None, None, None)

        try:
            manager.model = unpickle_from(manager.model_location)
        except:
            raise Exception(
                "Could not find a valid Mach model at the path:"
                f" {manager.model_location}"
            )

        manager.intro_source = DocumentDataSource.load(path=manager.intro_source_folder)
        manager.train_source = DocumentDataSource.load(path=manager.train_source_folder)
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
            intro_source=self.intro_source,
            train_source=self.train_source,
            tracker=self.tracker,
        )
