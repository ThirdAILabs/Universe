import json
from pathlib import Path

from ..defaults import training_arguments_for_scratch, training_arguments_from_base
from ..documents import DocumentDataSource
from ..utils import unpickle_from
from .checkpoint_config import NDBCheckpointConfig, TrainingDataCheckpointManager
from .training_callback import TrainingProgressManager
from .training_progress_tracker import NeuralDbProgressTracker, IntroState, TrainState


class TrainingProgressManagerFactory:
    @staticmethod
    def make_modelwise_checkpoint_configs_from_config(
        config: NDBCheckpointConfig, number_models
    ):
        """
        We maintain a checkpoint config for each Mach model in the Mixture while training. This is designed so that Mach models can maintain their training state independent of their Mixture which is necessary for distributed training.
        """
        if config:
            return [
                NDBCheckpointConfig(
                    config.checkpoint_dir / str(model_id),
                    config.resume_from_checkpoint,
                    config.checkpoint_interval,
                )
                for model_id in range(number_models)
            ]
        else:
            return [None] * number_models

    @staticmethod
    def make_resumed_training_progress_manager(
        original_mach_model,
        checkpoint_config: NDBCheckpointConfig,
    ) -> TrainingProgressManager:
        """
        Given a checkpoint, we will make a save load manager that will load the model, data sources, tracker.
        """
        assert checkpoint_config.checkpoint_dir != None

        save_load_manager = TrainingDataCheckpointManager.load(
            checkpoint_dir=checkpoint_config.checkpoint_dir
        )
        # We need to update the passed model with the state of the loaded model. Since, we need a model reference in the save_load_config as well, we update the model reference there as well.
        original_mach_model.reset_model(save_load_manager.model)
        save_load_manager.model = original_mach_model
        training_progress_manager = TrainingProgressManager(
            tracker=save_load_manager.tracker,
            save_load_config=save_load_manager,
            makes_checkpoint=True,
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
        checkpoint_config: NDBCheckpointConfig,
    ):
        intro_state = IntroState(
            num_buckets_to_sample=num_buckets_to_sample,
            fast_approximation=fast_approximation,
            override_number_classes=override_number_classes,
            is_insert_completed=False,
        )

        if model.model is None:
            train_args = training_arguments_for_scratch(train_documents.size)
        else:
            train_args = training_arguments_from_base(train_documents.size)

        train_state = TrainState(
            max_in_memory_batches=max_in_memory_batches,
            current_epoch_number=0,
            is_training_completed=not should_train,
            **train_args
        )

        tracker = NeuralDbProgressTracker(
            intro_state=intro_state, train_state=train_state, vlc_config=variable_length
        )

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
            checkpoint_interval=(
                checkpoint_config.checkpoint_interval if checkpoint_config else 1
            ),
        )

        if not should_train:
            training_progress_manager.tracker.is_training_completed = True

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
        checkpoint_config: NDBCheckpointConfig,
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
