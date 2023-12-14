import json
import os
import pickle
from pathlib import Path
import pandas as pd
from thirdai import bolt

from .training_progress_tracker import NeuralDbProgressTracker
from .checkpoint_config import CheckpointConfig
from ..utils import pickle_to, unpickle_from
from ..documents import DocumentDataSource


class TrainingProgressManager(bolt.train.callbacks.Callback):
    def __init__(
        self,
        tracker: NeuralDbProgressTracker,
        checkpoint_config: CheckpointConfig = None,
        neuraldb_mach_model=None,
        intro_source=None,
        train_source=None,
        is_resumed=False,
    ):
        super().__init__()
        self.tracker = tracker
        self.checkpoint_config = checkpoint_config
        self._is_resumed = is_resumed

        self._neuraldb_mach_model = neuraldb_mach_model
        self._intro_source = intro_source
        self._train_source = train_source

        if self.checkpoint_config:
            os.makedirs(
                os.path.join(self.checkpoint_config.checkpoint_dir),
                exist_ok=True,
            )

    def on_epoch_end(self):
        self.tracker.current_epoch_number += 1
        self.checkpoint_tracker()
        self.checkpoint_model()

    @property
    def neuraldb_mach_model(self):
        return self._neuraldb_mach_model

    @neuraldb_mach_model.setter
    def neuraldb_mach_model(self, model):
        self._neuraldb_mach_model = model

    @property
    def is_resumed(self):
        return self._is_resumed

    @property
    def intro_source(self):
        return self.load_intro_source()

    @property
    def train_source(self):
        return self.load_train_source()

    def checkpoint_model(self):
        if self._neuraldb_mach_model and self.checkpoint_config:
            pickle_to(
                obj=self._neuraldb_mach_model,
                filepath=self.checkpoint_config.neuraldb_model_checkpoint_location,
            )

    def checkpoint_sources(self):
        if self.checkpoint_config:
            if self._intro_source:
                dataframe = self._intro_source.dataframe()
                dataframe.to_csv(
                    self.checkpoint_config.intro_source_checkpoint_location,
                    index=False,
                )
                with open(
                    self.checkpoint_config.intro_source_arguments_location, "w"
                ) as f:
                    json.dump(self._intro_source.initialization_args(), f, indent=4)

            if self._train_source:
                dataframe = self._train_source.dataframe()
                dataframe.to_csv(
                    self.checkpoint_config.train_source_checkpoint_location, index=False
                )
                with open(
                    self.checkpoint_config.train_source_arguments_location, "w"
                ) as f:
                    json.dump(self._train_source.initialization_args(), f, indent=4)

    def checkpoint_tracker(self):
        if self.checkpoint_config:
            with open(self.checkpoint_config.intro_args_checkpoint_location, "w") as f:
                json.dump(self.tracker.introduce_arguments(), f, indent=4)

            with open(self.checkpoint_config.train_args_checkpoint_location, "w") as f:
                json.dump(self.tracker.training_arguments(), f, indent=4)

            with open(self.checkpoint_config.train_state_checkpoint_location, "w") as f:
                json.dump(self.tracker.training_state(), f, indent=4)

    def full_checkpoint(self):
        self.checkpoint_model()
        self.checkpoint_sources()
        self.checkpoint_tracker()

    def load_model(self):
        if self.checkpoint_config and not self.neuraldb_mach_model:
            try:
                neural_db_mach_model = unpickle_from(
                    filepath=self.checkpoint_config.neuraldb_model_checkpoint_location
                )
                self.neuraldb_mach_model = neural_db_mach_model
            except:
                print(
                    "No Model found in the checkpoint directory"
                    f" {self.checkpoint_config.checkpoint_dir}"
                )
                self.neuraldb_mach_model = None
        return self.neuraldb_mach_model

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
        if not self._intro_source:
            intro_source = TrainingProgressManager.load_source(
                arguments_location=self.tracker.intro_source_arguments_location,
                checkpoint_location=self.tracker.intro_source_checkpoint_location,
            )
            self._intro_source = intro_source
        return self._intro_source

    def load_train_source(self):
        if not self._train_source:
            train_source = TrainingProgressManager.load_source(
                arguments_location=self.tracker.train_source_arguments_location,
                checkpoint_location=self.tracker.train_source_checkpoint_location,
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
    def is_train_completed(self):
        return self.tracker.is_train_completed

    @is_train_completed.setter
    def is_train_completed(self, value):
        self.tracker.is_train_completed = value
