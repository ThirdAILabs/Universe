from __future__ import annotations

from thirdai import bolt

from ..documents import DocumentDataSource
from ..mach_defaults import (
    training_arguments_from_base,
    training_arguments_from_scratch,
)
from .checkpoint_config import CheckpointConfig
from .training_data_manager import TrainingDataManager
from .training_progress_tracker import IntroState, NeuralDbProgressTracker, TrainState


class TrainingProgressCallback(bolt.train.callbacks.Callback):
    def __init__(self, training_progress_manager: TrainingProgressManager):
        super().__init__()
        self.training_progress_manager = training_progress_manager

    def on_epoch_end(self):
        self.training_progress_manager.complete_epoch()


class TrainingProgressManager:
    """
    TrainingProgressManager class is used to maintain the checkpoint while training and is the source of truth for inserting documents into the current model object.

    This is designed the way it is to make sure that that we have identical function calls irrespective of whether we're resuming from a checkpoint/using checkpointing/doing no checkpointing. By making our training progress manager the source of truth for all training related variables/objects, we effectively offload the task of maintaining training state and checkpointing to the manager. And the manager internally decides when it should save what objects.

    Another reason to explain this unified design is that if we have seperate calls for indexing with/out checkpoint, we have to be sure that making changes in one does not break the other.
    """

    def __init__(
        self,
        tracker: NeuralDbProgressTracker,
        save_load_manager: TrainingDataManager,
        makes_checkpoint: bool,
        checkpoint_interval: int = 1,
    ):
        self.tracker = tracker
        self.save_load_manager = save_load_manager

        self.makes_checkpoint = makes_checkpoint
        self.checkpoint_interval = checkpoint_interval

        if self.makes_checkpoint:
            # Backup config saves into a different directory but all other model, tracker, source references remain the same as save_load_manager
            self.backup_config = save_load_manager.copy_with_new_dir(
                new_directory=save_load_manager.checkpoint_dir / "backup"
            )

    def complete_epoch(self):
        self.tracker.current_epoch_number += 1
        if self.tracker.current_epoch_number % self.checkpoint_interval == 0:
            if self.makes_checkpoint:
                self.checkpoint_without_sources()

    @property
    def intro_source(self) -> DocumentDataSource:
        return self.save_load_manager.intro_source

    @property
    def train_source(self) -> DocumentDataSource:
        return self.save_load_manager.train_source

    def make_preindexing_checkpoint(self):
        # Before starting indexing, we need to save all the resources (intro source, train source, model, tracker)
        # to be able to resume.
        if not self.makes_checkpoint:
            return
        self.save_load_manager.save()

    def training_complete(self):
        # Updates the tracker state by marking training as completed and saves the resources (tracker and model)
        # if makes_checkpoint is True.
        self.tracker.training_complete()
        if not self.makes_checkpoint:
            return
        self.checkpoint_without_sources()
        self.backup_config.delete_checkpoint()

    def insert_complete(self):
        # Updates the tracker state by marking insert as completed and saves the resources (tracker and model)
        # if makes_checkpoint is True.
        self.tracker.insert_complete()
        if not self.makes_checkpoint:
            return
        self.checkpoint_without_sources()

    def checkpoint_without_sources(self):
        # First save the model in the backup directory. Once the resources have been successfully saved,
        # we can move them to their intended checkpoint location. We only need to maintain backups of the
        # model and the tracker because other resources (intro and train source) are never modified.
        self.backup_config.save_without_sources()
        TrainingDataManager.update_model_and_tracker_from_backup(
            backup_config=self.backup_config, target_config=self.save_load_manager
        )

    def delete_backup(self):
        self.backup_config.delete_checkpoint()

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

    @staticmethod
    def from_scratch(
        model,
        intro_documents,
        train_documents,
        should_train,
        fast_approximation,
        num_buckets_to_sample,
        max_in_memory_batches,
        override_number_classes,
        variable_length,
        checkpoint_config: CheckpointConfig,
        **kwargs,
    ) -> TrainingProgressManager:
        intro_state = IntroState(
            num_buckets_to_sample=num_buckets_to_sample,
            fast_approximation=fast_approximation,
            override_number_classes=override_number_classes,
            is_insert_completed=False,
        )

        if model.model is None:
            train_args = training_arguments_from_scratch(train_documents.size)
        else:
            train_args = training_arguments_from_base(train_documents.size)

        train_args["batch_size"] = kwargs.get("batch_size", None)
        train_args["learning_rate"] = kwargs.get(
            "learning_rate", train_args["learning_rate"]
        )
        train_args["min_epochs"] = kwargs.get("epochs", train_args["min_epochs"])
        train_args["max_epochs"] = kwargs.get("epochs", train_args["max_epochs"])

        train_args["freeze_after_epoch"] = kwargs.get(
            "freeze_after_epoch", train_args["max_epochs"] - 1
        )
        train_args["freeze_after_acc"] = kwargs.get(
            "freeze_after_acc", 0.80 if "freeze_after_epoch" not in kwargs else 1
        )
        train_args["balancing_samples"] = kwargs.get("balancing_samples", False)
        train_args["semantic_enhancement"] = kwargs.get("semantic_enhancement", False)
        train_args["semantic_model_cache_dir"] = kwargs.get(
            "semantic_model_cache_dir", ".cache/neural_db_semantic_model"
        )

        train_state = TrainState(
            max_in_memory_batches=max_in_memory_batches,
            current_epoch_number=0,
            is_training_completed=not should_train,
            **train_args,
        )

        tracker = NeuralDbProgressTracker(
            intro_state=intro_state, train_state=train_state, vlc_config=variable_length
        )

        save_load_manager = TrainingDataManager(
            checkpoint_dir=(
                checkpoint_config.checkpoint_dir if checkpoint_config else None
            ),
            model=model,
            intro_source=intro_documents,
            train_source=train_documents,
            tracker=tracker,
        )

        training_progress_manager = TrainingProgressManager(
            tracker=tracker,
            save_load_manager=save_load_manager,
            makes_checkpoint=True if checkpoint_config else False,
            checkpoint_interval=(
                checkpoint_config.checkpoint_interval if checkpoint_config else 1
            ),
        )

        if not should_train:
            training_progress_manager.tracker.is_training_completed = True

        return training_progress_manager

    @staticmethod
    def from_checkpoint(
        original_mach_model,
        checkpoint_config: CheckpointConfig,
    ) -> TrainingProgressManager:
        """
        Given a checkpoint, we will make a save load manager that will load the model, data sources, tracker.
        """
        assert checkpoint_config.checkpoint_dir != None

        save_load_manager = TrainingDataManager.load(
            checkpoint_dir=checkpoint_config.checkpoint_dir
        )
        # We need to update the passed model with the state of the loaded model. Since, we need a model reference in the save_load_manager as well, we update the model reference there too.
        original_mach_model.reset_model(save_load_manager.model)
        save_load_manager.model = original_mach_model
        training_progress_manager = TrainingProgressManager(
            tracker=save_load_manager.tracker,
            save_load_manager=save_load_manager,
            makes_checkpoint=True,
            checkpoint_interval=checkpoint_config.checkpoint_interval,
        )
        return training_progress_manager
