from thirdai import bolt

from ..documents import DocumentDataSource
from .checkpoint_config import TrainingDataCheckpointManager
from .training_progress_tracker import NeuralDbProgressTracker


class TrainingProgressManager(bolt.train.callbacks.Callback):  # type: ignore
    def __init__(
        self,
        tracker: NeuralDbProgressTracker,
        save_load_config: TrainingDataCheckpointManager,
        makes_checkpoint: bool,
        is_resumed: bool,
        checkpoint_interval: int = 1,
    ):
        super().__init__()
        self.tracker = tracker
        self.save_load_config = save_load_config

        self.makes_checkpoint = makes_checkpoint
        self._is_resumed = is_resumed
        self.checkpoint_interval = checkpoint_interval

        if self.makes_checkpoint:
            # Backup config saves into a different directory but all other model, tracker, source references remain the same as save_load_config
            self.backup_config = save_load_config.copy_with_new_dir(
                new_directory=save_load_config.checkpoint_dir / ".temp"
            )

    def on_epoch_end(self):
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
        if self._is_resumed or not self.makes_checkpoint:
            return
        self.save_load_config.save()

    def training_complete(self):
        if self.is_training_completed or not self.makes_checkpoint:
            return
        self.is_training_completed = True
        self.checkpoint_without_sources()

    def insert_complete(self):
        if self.is_insert_completed or not self.makes_checkpoint:
            return
        self.is_insert_completed = True
        self.checkpoint_without_sources()

    def checkpoint_without_sources(self):
        self.backup_config.save_without_sources()
        TrainingDataCheckpointManager.update_model_and_tracker_from_backup(
            backup_config=self.backup_config, target_config=self.save_load_config
        )

    def set_pretraining_arguments(
        self,
        fast_approximation: bool,
        num_buckets_to_sample: int,
        override_number_classes: int,
        vlc_config,
    ):
        self.tracker.fast_approximation = fast_approximation
        self.tracker.num_buckets_to_sample = num_buckets_to_sample
        self.tracker.override_number_classes = override_number_classes
        self.tracker.vlc_config = vlc_config

    def set_training_arguments(
        self,
        learning_rate: float,
        min_epochs: int,
        max_epochs: int,
        freeze_before_train: bool,
    ):
        self.tracker.learning_rate = learning_rate
        self.tracker.min_epochs = min_epochs
        self.tracker.max_epochs = max_epochs
        self.tracker.freeze_before_train = freeze_before_train

    @property
    def is_insert_completed(self):
        return self.tracker.is_insert_completed

    @is_insert_completed.setter
    def is_insert_completed(self, value):
        self.tracker.is_insert_completed = value

    @property
    def is_training_completed(self):
        return self.tracker.is_training_completed

    @is_training_completed.setter
    def is_training_completed(self, value):
        self.tracker.is_training_completed = value

    @property
    def min_epochs(self):
        return max(self.tracker.min_epochs - self.tracker.current_epoch_number, 0)

    @property
    def max_epochs(self):
        return self.tracker.max_epochs - self.tracker.current_epoch_number

    def get_training_arguments(self):
        return {
            "min_epochs": self.min_epochs,
            "max_epochs": self.max_epochs,
            "learning_rate": self.tracker.learning_rate,
            "freeze_before_train": self.tracker.freeze_before_train,
            "max_in_memory_batches": self.tracker.max_in_memory_batches,
            "variable_length": self.tracker.vlc_config,
            "training_progress_manager": self,
        }
