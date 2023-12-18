import json
import os
from pathlib import Path
from typing import Union

from thirdai import bolt

from ..documents import DocumentDataSource
from ..utils import pickle_to, unpickle_from
from .checkpoint_config import DirConfig
from .training_progress_tracker import NeuralDbProgressTracker


class TrainingProgressManager(bolt.train.callbacks.Callback):  # type: ignore
    def __init__(
        self,
        tracker: NeuralDbProgressTracker,
        dir_config: Union[DirConfig, None] = None,
        neuraldb_mach_model=None,
        intro_source=None,
        train_source=None,
        checkpoint_interval: int = 1,
    ):
        super().__init__()
        self.tracker = tracker
        self.dir_config = dir_config

        self._neuraldb_mach_model = neuraldb_mach_model
        self._intro_source = intro_source
        self._train_source = train_source
        self.checkpoint_interval = checkpoint_interval

        if self.dir_config:
            os.makedirs(
                os.path.join(self.dir_config.checkpoint_dir),
                exist_ok=True,
            )

    def on_epoch_end(self):
        self.tracker.current_epoch_number += 1
        if self.tracker.current_epoch_number % self.checkpoint_interval == 0:
            self.checkpoint_tracker()
            self.checkpoint_model()

    def on_train_end(self):
        self.checkpoint_tracker()
        self.checkpoint_model()

    @property
    def neuraldb_mach_model(self):
        if self._neuraldb_mach_model is None:
            self._neuraldb_mach_model = self.load_model()

        if self._neuraldb_mach_model is None:
            raise Exception("Invalid NeuralDB Model in Training Progress Manager.")
        return self._neuraldb_mach_model

    @neuraldb_mach_model.setter
    def neuraldb_mach_model(self, model):
        self._neuraldb_mach_model = model

    @property
    def intro_source(self):
        return self.load_intro_source()

    @property
    def train_source(self):
        return self.load_train_source()

    def checkpoint_model(self):
        # We do not use the property attribute for accessing the underlying neuraldb_mach_model because calling it also loads it.
        if self._neuraldb_mach_model and self.dir_config:
            print("Checkpoint Model")
            pickle_to(
                obj=self._neuraldb_mach_model,
                filepath=self.dir_config.neuraldb_model_checkpoint_location,
            )

    def checkpoint_sources(self):
        if self.dir_config:
            if self._intro_source:
                print("Checkpoint Intro Source")
                dataframe = self._intro_source.dataframe()
                dataframe.to_csv(
                    self.dir_config.intro_source_checkpoint_location,
                    index=False,
                )
                with open(self.dir_config.intro_source_arguments_location, "w") as f:
                    json.dump(self._intro_source.initialization_args(), f, indent=4)

            if self._train_source:
                print("Checkpoint Train source")
                dataframe = self._train_source.dataframe()
                dataframe.to_csv(
                    self.dir_config.train_source_checkpoint_location, index=False
                )
                with open(self.dir_config.train_source_arguments_location, "w") as f:
                    json.dump(self._train_source.initialization_args(), f, indent=4)

    def checkpoint_tracker(self):
        if self.dir_config:
            print("Checkpointing Tracker")
            with open(self.dir_config.intro_args_checkpoint_location, "w") as f:
                json.dump(self.tracker.introduce_arguments(), f, indent=4)

            with open(self.dir_config.train_args_checkpoint_location, "w") as f:
                json.dump(self.tracker.training_arguments(), f, indent=4)

            with open(self.dir_config.train_state_checkpoint_location, "w") as f:
                json.dump(self.tracker.training_state(), f, indent=4)

    def full_checkpoint(self):
        self.checkpoint_model()
        self.checkpoint_sources()
        self.checkpoint_tracker()

    def make_insert_checkpoint(self):
        if self.is_insert_completed:
            return

        self.is_insert_completed = True
        self.full_checkpoint()

    def load_model(self):
        if self.dir_config and not self._neuraldb_mach_model:
            try:
                neural_db_mach_model = unpickle_from(
                    filepath=self.dir_config.neuraldb_model_checkpoint_location
                )
                self._neuraldb_mach_model = neural_db_mach_model
                print("loaded a valid model")
            except:
                raise Exception(
                    "Could not model. Ensure that the model checkpoint is valid"
                )
        return self._neuraldb_mach_model

    @staticmethod
    def load_source(
        arguments_location: Path, checkpoint_location: Path
    ) -> DocumentDataSource:
        with open(arguments_location) as f:
            args = json.load(f)
        source = DocumentDataSource.load_from_dataframe(
            csv_path=checkpoint_location,
            id_column=args["id_column"],
            strong_column=args["strong_column"],
            weak_column=args["weak_column"],
        )
        return source

    def load_intro_source(self):
        if not self._intro_source and self.dir_config:
            intro_source = TrainingProgressManager.load_source(
                arguments_location=self.dir_config.intro_source_arguments_location,
                checkpoint_location=self.dir_config.intro_source_checkpoint_location,
            )
            self._intro_source = intro_source
        return self._intro_source

    def load_train_source(self):
        if not self._train_source and self.dir_config:
            train_source = TrainingProgressManager.load_source(
                arguments_location=self.dir_config.train_source_arguments_location,
                checkpoint_location=self.dir_config.train_source_checkpoint_location,
            )
            self._train_source = train_source
        return self._train_source

    def set_introduce_arguments(
        self,
        fast_approximation: bool,
        num_buckets_to_sample: int,
        override_number_classes: int,
    ):
        self.tracker.fast_approximation = fast_approximation
        self.tracker.num_buckets_to_sample = num_buckets_to_sample
        self.tracker.override_number_classes = override_number_classes

    def set_training_arguments(
        self,
        learning_rate: float,
        min_epochs: int,
        max_epochs: int,
        freeze_before_train: bool,
    ):
        if self.tracker.learning_rate is None:
            self.tracker.learning_rate = learning_rate
        if self.tracker.min_epochs is None:
            self.tracker.min_epochs = min_epochs
        if self.tracker.max_epochs is None:
            self.tracker.max_epochs = max_epochs
        if self.tracker.freeze_before_train is None:
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
        return self.tracker.min_epochs - self.tracker.current_epoch_number

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
            "training_progress_manager": self,
        }
