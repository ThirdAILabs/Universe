import json
from pathlib import Path

from ..utils import unpickle_from
from .checkpoint_config import CheckpointConfig, DirectoryConfig
from .training_callback import TrainingProgressManager
from .training_progress_tracker import (
    IntroState,
    NeuralDbProgressTracker,
    TrainState,
)


class TrainingProgressManagerFactory:
    @staticmethod
    def make_modelwise_checkpoint_configs_from_config(
        config: CheckpointConfig, number_models
    ):
        """
        We maintain a checkpoint config for each Mach model in the Mixture while training. This is designed so that Mach models can maintain their training state independent of their Mixture which is necessary for distributed training.
        """
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
    ) -> DirectoryConfig:
        return DirectoryConfig(checkpoint_dir=checkpoint_config.checkpoint_dir)

    @staticmethod
    def load_tracker_from_checkpoint(
        checkpoint_config: CheckpointConfig,
    ) -> NeuralDbProgressTracker:
        dir_config = TrainingProgressManagerFactory.make_dir_config_for_checkpoint(
            checkpoint_config=checkpoint_config
        )
        dir_config.assert_checkpoint_source_exists()

        with open(dir_config.train_and_intro_state_checkpoint_location, "r") as f:
            params = json.load(f)
        vlc_config = unpickle_from(filepath=dir_config.vlc_path)

        return NeuralDbProgressTracker.load(
            arguments_json=params, vlc_config=vlc_config
        )

    @staticmethod
    def make_default_tracker() -> NeuralDbProgressTracker:
        return NeuralDbProgressTracker()

    @staticmethod
    def make_resumed_training_progress_manager(
        checkpoint_config: CheckpointConfig,
    ) -> TrainingProgressManager:
        """
        Given a checkpoint, we will make a training manager that will load the model, data sources, tracker, and resume training.
        """
        assert checkpoint_config.checkpoint_dir != None

        tracker = TrainingProgressManagerFactory.load_tracker_from_checkpoint(
            checkpoint_config
        )

        dir_config = TrainingProgressManagerFactory.make_dir_config_for_checkpoint(
            checkpoint_config=checkpoint_config
        )

        # We set neuraldb_mach_model, intro_source, train_source as None so that calling load on the training manager will initialize
        training_progress_manager = TrainingProgressManager(
            tracker=tracker,
            dir_config=dir_config,
            neuraldb_mach_model=None,
            intro_source=None,
            train_source=None,
            is_resumed=True,
        )
        # Calling load on the training manager also updates the internal class attributes.
        training_progress_manager.load_model()
        training_progress_manager.load_intro_source()
        training_progress_manager.load_train_source()

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
        variable_length,
        checkpoint_config: CheckpointConfig,
    ):
        tracker = TrainingProgressManagerFactory.make_default_tracker()

        if checkpoint_config:
            dir_config = TrainingProgressManagerFactory.make_dir_config_for_checkpoint(
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

        training_progress_manager.set_pretraining_arguments(
            fast_approximation=fast_approximation,
            num_buckets_to_sample=num_buckets_to_sample,
            override_number_classes=override_number_classes,
            vlc_config=variable_length,
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
        variable_length,
        checkpoint_config: CheckpointConfig,
    ):

        if checkpoint_config and checkpoint_config.resume_from_checkpoint:
            return (
                TrainingProgressManagerFactory.make_resumed_training_progress_manager(
                    checkpoint_config=checkpoint_config
                )
            )
        else:
            return TrainingProgressManagerFactory.make_training_manager_scratch(
                model=model,
                intro_documents=intro_documents,
                train_documents=train_documents,
                should_train=should_train,
                fast_approximation=fast_approximation,
                num_buckets_to_sample=num_buckets_to_sample,
                max_in_memory_batches=max_in_memory_batches,
                override_number_classes=override_number_classes,
                variable_length=variable_length,
                checkpoint_config=checkpoint_config,
            )
