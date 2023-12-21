from __future__ import annotations
from thirdai import bolt

from ..documents import DocumentDataSource
from .checkpoint_config import TrainingDataCheckpointManager
from .training_progress_tracker import NeuralDbProgressTracker


class TrainingProgressCallback(bolt.train.callbacks.Callback):  # type: ignore
    def __init__(self, training_progress_manager: TrainingProgressManager):
        super().__init__()
        self.training_progress_manager = training_progress_manager

    def on_epoch_end(self):
        self.training_progress_manager.complete_epoch()


class TrainingProgressManager(bolt.train.callbacks.Callback):  # type: ignore
    def __init__(
        self,
        tracker: NeuralDbProgressTracker,
        save_load_config: TrainingDataCheckpointManager,
        makes_checkpoint: bool,
        checkpoint_interval: int = 1,
    ):
        super().__init__()
        self.tracker = tracker
        self.save_load_config = save_load_config

        self.makes_checkpoint = makes_checkpoint
        self.checkpoint_interval = checkpoint_interval

        if self.makes_checkpoint:
            # Backup config saves into a different directory but all other model, tracker, source references remain the same as save_load_config
            self.backup_config = save_load_config.copy_with_new_dir(
                new_directory=save_load_config.checkpoint_dir / ".temp"
            )

    def complete_epoch(self):
        self.tracker.current_epoch_number += 1
        if self.tracker.current_epoch_number % self.checkpoint_interval == 0:
            if self.makes_checkpoint:
                self.checkpoint_without_sources()

    @property
    def intro_source(self) -> DocumentDataSource:
        return self.save_load_config.intro_source

    @property
    def train_source(self) -> DocumentDataSource:
        return self.save_load_config.train_source

    def make_preindexing_checkpoint(self):
        if not self.makes_checkpoint:
            return
        self.save_load_config.save()

    def training_complete(self):
        if self.is_training_completed or not self.makes_checkpoint:
            return
        self.tracker.training_complete()
        self.checkpoint_without_sources()

    def insert_complete(self):
        if self.is_insert_completed or not self.makes_checkpoint:
            return
        self.tracker.insert_complete()
        self.checkpoint_without_sources()

    def checkpoint_without_sources(self):
        self.backup_config.save_without_sources()
        TrainingDataCheckpointManager.update_model_and_tracker_from_backup(
            backup_config=self.backup_config, target_config=self.save_load_config
        )

    @property
    def is_insert_completed(self):
        return self.tracker.is_insert_completed

    @property
    def is_training_completed(self):
        return self.tracker.is_training_completed

    def training_arguments(self):
        return self.tracker.training_arguments()

    def introduce_arguments(self):
        return self.tracker.introduce_arguments()
