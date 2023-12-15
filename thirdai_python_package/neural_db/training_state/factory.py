import json
from pathlib import Path

from .checkpoint_config import DirConfig, CheckpointConfig
from .training_callback import TrainingProgressManager
from .training_progress_tracker import (
    IntroConfig,
    NeuralDbProgressTracker,
    TrainingConfig,
)


class Factory:
    @staticmethod
    def make_modelwise_checkpoint_configs_from_config(
        config: CheckpointConfig, number_models
    ):
        if config:
            return [
                CheckpointConfig(
                    config.checkpoint_dir / str(model_id),
                    config.resume_from_checkpoint,
                    config.checkpoint_interval,
                )
                for model_id in range(number_models)
            ]
        else:
            return [None] * number_models

    @staticmethod
    def make_dir_config_for_checkpoint(
        checkpoint_config: CheckpointConfig,
    ) -> DirConfig:
        return DirConfig(checkpoint_dir=checkpoint_config.checkpoint_dir)

    @staticmethod
    def load_tracker_from_checkpoint(
        checkpoint_config: CheckpointConfig,
    ) -> NeuralDbProgressTracker:
        # We first make the dir config that tells us the locations
        # of where the tracking data is stored
        dir_config = Factory.make_dir_config_for_checkpoint(
            checkpoint_config=checkpoint_config
        )
        dir_config.assert_checkpoint_source_exists()

        tracker = NeuralDbProgressTracker()
        tracker._intro_config = IntroConfig.load(
            dir_config.intro_args_checkpoint_location
        )
        tracker._train_config = TrainingConfig.load(
            training_args_path=dir_config.train_args_checkpoint_location,
            training_state_path=dir_config.train_state_checkpoint_location,
        )
        return tracker

    @staticmethod
    def make_default_tracker() -> NeuralDbProgressTracker:
        return NeuralDbProgressTracker()

    @staticmethod
    def make_resumed_training_progress_manager(
        checkpoint_config: CheckpointConfig,
    ) -> TrainingProgressManager:
        assert checkpoint_config.checkpoint_dir != None

        tracker = Factory.load_tracker_from_checkpoint(checkpoint_config)

        dir_config = Factory.make_dir_config_for_checkpoint(
            checkpoint_config=checkpoint_config
        )

        training_progress_manager = TrainingProgressManager(
            tracker=tracker,
            dir_config=dir_config,
            neuraldb_mach_model=None,
            intro_source=None,
            train_source=None,
        )
        return training_progress_manager

    @staticmethod
    def make_training_manager_scratch(
        model,
        intro_documents,
        train_documents,
        should_train,
        fast_approximation,
        num_buckets_to_sample,
        max_in_memory_batches,
        override_number_classes,
        checkpoint_config: CheckpointConfig,
    ):
        tracker = Factory.make_default_tracker()

        if checkpoint_config:
            dir_config = Factory.make_dir_config_for_checkpoint(
                checkpoint_config=checkpoint_config
            )
        else:
            dir_config = None

        training_progress_manager = TrainingProgressManager(
            tracker=tracker,
            dir_config=dir_config,
            neuraldb_mach_model=model,
            intro_source=intro_documents,
            train_source=train_documents,
            checkpoint_interval=(
                checkpoint_config.checkpoint_interval if checkpoint_config else 1
            ),
        )

        training_progress_manager.set_introduce_arguments(
            fast_approximation=fast_approximation,
            num_buckets_to_sample=num_buckets_to_sample,
            override_number_classes=override_number_classes,
        )

        if not should_train:
            training_progress_manager.tracker.is_training_completed = True
        training_progress_manager.tracker.max_in_memory_batches = max_in_memory_batches

        return training_progress_manager

    @staticmethod
    def make_training_progress_manager(
        model,
        intro_documents,
        train_documents,
        should_train,
        fast_approximation,
        num_buckets_to_sample,
        max_in_memory_batches,
        override_number_classes,
        checkpoint_config: CheckpointConfig,
    ):

        if checkpoint_config and checkpoint_config.resume_from_checkpoint:
            return Factory.make_resumed_training_progress_manager(
                checkpoint_config=checkpoint_config
            )
        else:
            return Factory.make_training_manager_scratch(
                model=model,
                intro_documents=intro_documents,
                train_documents=train_documents,
                should_train=should_train,
                fast_approximation=fast_approximation,
                num_buckets_to_sample=num_buckets_to_sample,
                max_in_memory_batches=max_in_memory_batches,
                override_number_classes=override_number_classes,
                checkpoint_config=checkpoint_config,
            )
