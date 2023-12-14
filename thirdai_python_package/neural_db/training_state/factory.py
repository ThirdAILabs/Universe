import json
from pathlib import Path

from .checkpoint_config import CheckpointConfig
from .training_callback import TrainingProgressManager
from .training_progress_tracker import (
    IntroConfig,
    NeuralDbProgressTracker,
    TrainingConfig,
)


class Factory:
    def make_checkpoint_config(
        parent_checkpoint_dir: Path, model_id: int
    ) -> CheckpointConfig:
        return CheckpointConfig(
            parent_checkpoint_dir=parent_checkpoint_dir, model_id=model_id
        )

    def load_tracker_from_dir(
        parent_checkpoint_dir: Path, model_id: int
    ) -> NeuralDbProgressTracker:
        # We first load the checkpoint config that tells us the locations
        # of where the tracking data is stored
        checkpoint_config = CheckpointConfig(
            parent_checkpoint_dir=parent_checkpoint_dir, model_id=model_id
        )
        checkpoint_config.assert_checkpoint_source_exists()

        tracker = NeuralDbProgressTracker()
        tracker._intro_config = IntroConfig.load(
            checkpoint_config.intro_args_checkpoint_location
        )
        tracker._train_config = TrainingConfig.load(
            training_args_path=checkpoint_config.train_args_checkpoint_location,
            training_state_path=checkpoint_config.train_state_checkpoint_location,
        )
        return tracker

    def load_tracker_without_dir():
        return NeuralDbProgressTracker()

    def make_training_progress_manager(
        model,
        intro_documents,
        train_documents,
        should_train,
        fast_approximation,
        num_buckets_to_sample,
        max_in_memory_batches,
        override_number_classes,
        checkpoint_dir,
        model_id,
        resume_from_checkpoint,
    ):

        if resume_from_checkpoint:
            assert checkpoint_dir != None
            assert model_id != None

            tracker = Factory.load_tracker_from_dir(
                parent_checkpoint_dir=checkpoint_dir, model_id=model_id
            )
            checkpoint_config = Factory.make_checkpoint_config(
                parent_checkpoint_dir=checkpoint_dir, model_id=model_id
            )
            training_progress_manager = TrainingProgressManager(
                tracker=tracker,
                checkpoint_config=checkpoint_config,
                neuraldb_mach_model=None,
                intro_source=None,
                train_source=None,
                is_resumed=True,
            )
            return training_progress_manager
        else:
            tracker = Factory.load_tracker_without_dir()
            if checkpoint_dir and model_id is not None:
                checkpoint_config = Factory.make_checkpoint_config(
                    parent_checkpoint_dir=checkpoint_dir, model_id=model_id
                )
                print("We are using the checkpoint config")
            else:
                checkpoint_config = None
                print("We are using none config")

            training_progress_manager = TrainingProgressManager(
                tracker=tracker,
                checkpoint_config=checkpoint_config,
                neuraldb_mach_model=model,
                intro_source=intro_documents,
                train_source=train_documents,
                is_resumed=False,
            )
            training_progress_manager.set_introduce_arguments(
                fast_approximation=fast_approximation,
                num_buckets_to_sample=num_buckets_to_sample,
                override_number_classes=override_number_classes,
            )
            if not should_train:
                training_progress_manager.tracker.is_training_completed = True
            tracker.max_in_memory_batches = max_in_memory_batches

            return training_progress_manager
