import json
from pathlib import Path

from ..documents import DocumentDataSource
from ..utils import unpickle_from
from .checkpoint_config import CheckpointConfig, TrainingDataCheckpointManager
from .training_callback import TrainingProgressManager
from .training_progress_tracker import NeuralDbProgressTracker


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

    def make_save_load_manager_scratch(
        self,
        checkpoint_dir: Path,
        model,
        intro_source: DocumentDataSource,
        train_source: DocumentDataSource,
        tracker: NeuralDbProgressTracker,
    ):
        return TrainingDataCheckpointManager(
            checkpoint_dir=checkpoint_dir,
            model=model,
            intro_source=intro_source,
            train_source=train_source,
            tracker=tracker,
        )

    @staticmethod
    def make_save_load_manager_from_checkpoint(
        checkpoint_dir: Path,
    ) -> TrainingDataCheckpointManager:
        return TrainingDataCheckpointManager.load(checkpoint_dir=checkpoint_dir)

    @staticmethod
    def make_default_tracker() -> NeuralDbProgressTracker:
        return NeuralDbProgressTracker()

    @staticmethod
    def make_resumed_training_progress_manager(
        original_mach_model,
        checkpoint_config: CheckpointConfig,
    ) -> TrainingProgressManager:
        """
        Given a checkpoint, we will make a save load manager that will load the model, data sources, tracker.
        """
        assert checkpoint_config.checkpoint_dir != None

        save_load_manager = (
            TrainingProgressManagerFactory.make_save_load_manager_from_checkpoint(
                checkpoint_dir=checkpoint_config.checkpoint_dir
            )
        )
        # We need to update the passed model with the state of the loaded model. Since, we need a model reference in the save_load_config as well, we update the model reference there as well.
        original_mach_model.reset_model(save_load_manager.model)
        save_load_manager.model = original_mach_model
        training_progress_manager = TrainingProgressManager(
            tracker=save_load_manager.tracker,
            save_load_config=save_load_manager,
            makes_checkpoint=True,
            is_resumed=True,
            checkpoint_interval=checkpoint_config.checkpoint_interval,
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
        variable_length,
        checkpoint_config: CheckpointConfig,
    ):
        tracker = TrainingProgressManagerFactory.make_default_tracker()

        save_load_manager = TrainingDataCheckpointManager(
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
            save_load_config=save_load_manager,
            makes_checkpoint=True if checkpoint_config else False,
            is_resumed=False,
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
                    model, checkpoint_config=checkpoint_config
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
