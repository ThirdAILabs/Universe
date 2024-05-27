import json
from dataclasses import dataclass
from pathlib import Path
from typing import List, Union

from thirdai import data

from ..utils import pickle_to, unpickle_from


@dataclass
class SupervisedTrainState:

    is_training_completed: bool
    current_epoch_number: int
    learning_rate: float
    epochs: int
    batch_size: int
    max_in_memory_batches: int
    metrics: List[int]
    disable_finetunable_retriever: bool


@dataclass
class InsertTrainState:
    max_in_memory_batches: int
    current_epoch_number: int
    is_training_completed: bool
    learning_rate: float
    min_epochs: int
    max_epochs: int
    freeze_before_train: bool
    batch_size: int
    freeze_after_epoch: int
    freeze_after_acc: float
    balancing_samples: bool
    semantic_enhancement: bool
    semantic_model_cache_dir: str


class IntroState:
    def __init__(
        self,
        num_buckets_to_sample: int,
        fast_approximation: bool,
        override_number_classes: bool,
        is_insert_completed: bool,
        **kwargs,
    ):
        self.num_buckets_to_sample = num_buckets_to_sample
        self.fast_approximation = fast_approximation
        self.override_number_classes = override_number_classes
        self.is_insert_completed = is_insert_completed


class NeuralDbProgressTracker:
    """
    This class will be used to track the current training status of a NeuralDB Mach Model.
    The training state needs to be updated constantly while a model is being trained and
    hence, this should ideally be used inside a callback.

    Given the NeuralDbProgressTracker of the model and the data sources, we should be able to resume the training.
    """

    def __init__(self, train_state: Union[SupervisedTrainState, InsertTrainState]):
        # These are training arguments and are updated while the training is in progress
        self._train_state = train_state

    @property
    def is_insert_completed(self):
        raise NotImplementedError(
            "Method 'is_insert_completed' not implemented for class NeuralDBProgressTracker."
        )

    @is_insert_completed.setter
    def is_insert_completed(self, is_insert_completed: bool):
        raise NotImplementedError(
            "Setter Method 'is_insert_completed' not implemented for class NeuralDBProgressTracker."
        )

    def insert_complete(self):
        raise NotImplementedError(
            "Method 'insert_complete' not implemented for class NeuralDBProgressTracker."
        )

    def introduce_arguments(self):
        raise NotImplementedError(
            "Method 'introduce_arguments' not implemented for class NeuralDBProgressTracker."
        )

    @property
    def is_training_completed(self):
        return self._train_state.is_training_completed

    @is_training_completed.setter
    def is_training_completed(self, is_training_completed: bool):
        if isinstance(is_training_completed, bool):
            self._train_state.is_training_completed = is_training_completed
        else:
            raise TypeError("Can set the property only with a bool")

    @property
    def current_epoch_number(self):
        return self._train_state.current_epoch_number

    @current_epoch_number.setter
    def current_epoch_number(self, current_epoch_number: int):
        if isinstance(current_epoch_number, int):
            self._train_state.current_epoch_number = current_epoch_number
        else:
            raise TypeError("Can set the property only with an int")

    def epoch_complete(self):
        self.current_epoch_number += 1

    def training_complete(self):
        if self.is_training_completed:
            raise Exception("Training has already been finished.")
        self.is_training_completed = True


class InsertProgressTracker(NeuralDbProgressTracker):
    def __init__(
        self,
        intro_state: IntroState,
        train_state: InsertTrainState,
        vlc_config: data.transformations.VariableLengthConfig,
    ):
        super().__init__(train_state=train_state)

        # These are the introduce state arguments and updated once the introduce document is done
        self._intro_state = intro_state

        # These are training arguments and are updated while the training is in progress
        self._train_state = train_state
        self.vlc_config = vlc_config

    @property
    def is_insert_completed(self):
        return self._intro_state.is_insert_completed

    @is_insert_completed.setter
    def is_insert_completed(self, is_insert_completed: bool):
        if isinstance(is_insert_completed, bool):
            self._intro_state.is_insert_completed = is_insert_completed
        else:
            raise TypeError("Can set the property only with a bool")

    def insert_complete(self):
        if self.is_insert_completed:
            raise Exception("Insert has already been finished.")
        self.is_insert_completed = True

    def introduce_arguments(self):
        return {
            "num_buckets_to_sample": self._intro_state.num_buckets_to_sample,
            "fast_approximation": self._intro_state.fast_approximation,
            "override_number_classes": self._intro_state.override_number_classes,
        }

    def training_arguments(self):
        min_epochs = (
            self._train_state.min_epochs - self._train_state.current_epoch_number
        )
        max_epochs = (
            self._train_state.max_epochs - self._train_state.current_epoch_number
        )
        freeze_after_epochs = (
            self._train_state.freeze_after_epoch
            - self._train_state.current_epoch_number
        )

        args = self._train_state.__dict__.copy()

        args["freeze_after_epochs"] = freeze_after_epochs
        args["min_epochs"] = min_epochs
        args["max_epochs"] = max_epochs

        args["variable_length"] = self.vlc_config

        return args

    def save(self, path: Path):
        path.mkdir(exist_ok=True, parents=True)
        with open(path / "tracker.json", "w") as f:
            json.dump(
                {
                    "intro_state": self._intro_state.__dict__,
                    "train_state": self._train_state.__dict__,
                },
                f,
                indent=4,
            )
        pickle_to(self.vlc_config, path / "vlc.config")

    @staticmethod
    def load(path: Path):
        with open(path / "tracker.json", "r") as f:
            args = json.load(f)

        vlc_config = unpickle_from(path / "vlc.config")

        return InsertProgressTracker(
            intro_state=IntroState(**args["intro_state"]),
            train_state=InsertTrainState(**args["train_state"]),
            vlc_config=vlc_config,
        )


class SupervisedProgressTracker(NeuralDbProgressTracker):
    def __init__(self, train_state: SupervisedTrainState):
        super().__init__(train_state=train_state)
        self._train_state = train_state

    def training_arguments(self):
        epochs = self._train_state.epochs - self.current_epoch_number
        args = self._train_state.__dict__.copy()
        args["epochs"] = epochs
        return args

    def save(self, path: Path):
        path.mkdir(exist_ok=True, parents=True)
        with open(path / "tracker.json", "w") as f:
            json.dump(
                self._train_state.__dict__,
                f,
                indent=4,
            )

    @staticmethod
    def load(path: Path):
        with open(path / "tracker.json", "r") as f:
            args = json.load(f)

        return SupervisedProgressTracker(train_state=SupervisedTrainState(**args))
