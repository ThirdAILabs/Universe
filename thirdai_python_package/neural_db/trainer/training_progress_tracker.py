import json
from pathlib import Path

from thirdai import data

from ..utils import pickle_to, unpickle_from


class TrainState:
    def __init__(
        self,
        max_in_memory_batches: int,
        current_epoch_number: int,
        is_training_completed: bool,
        learning_rate: float,
        min_epochs: int,
        max_epochs: int,
        freeze_before_train: bool,
        batch_size: int,
        freeze_after_epoch: int,
        freeze_after_acc: float,
        balancing_samples: bool,
        semantic_enhancement: bool,
        semantic_model_cache_dir: str,
        **kwargs,
    ):
        self.max_in_memory_batches = max_in_memory_batches
        self.current_epoch_number = current_epoch_number
        self.is_training_completed = is_training_completed
        self.learning_rate = learning_rate
        self.min_epochs = min_epochs
        self.max_epochs = max_epochs
        self.freeze_before_train = freeze_before_train
        self.batch_size = batch_size
        self.freeze_after_epoch = freeze_after_epoch
        self.freeze_after_acc = freeze_after_acc
        self.balancing_samples = balancing_samples
        self.semantic_enhancement = semantic_enhancement
        self.semantic_model_cache_dir = semantic_model_cache_dir


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

    def __init__(self, intro_state: IntroState, train_state: TrainState, vlc_config):
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

    def __dict__(self):
        return {
            "intro_state": self._intro_state.__dict__,
            "train_state": self._train_state.__dict__,
        }

    def insert_complete(self):
        if self.is_insert_completed:
            raise Exception("Insert has already been finished.")
        self.is_insert_completed = True

    def epoch_complete(self):
        self.current_epoch_number += 1

    def training_complete(self):
        if self.is_training_completed:
            raise Exception("Training has already been finished.")
        self.is_training_completed = True

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

    def introduce_arguments(self):
        return {
            "num_buckets_to_sample": self._intro_state.num_buckets_to_sample,
            "fast_approximation": self._intro_state.fast_approximation,
            "override_number_classes": self._intro_state.override_number_classes,
        }

    def save(self, path: Path):
        path.mkdir(exist_ok=True, parents=True)
        with open(path / "tracker.json", "w") as f:
            json.dump(self.__dict__(), f, indent=4)
        pickle_to(self.vlc_config, path / "vlc.config")

    @staticmethod
    def load(path: Path):
        with open(path / "tracker.json", "r") as f:
            args = json.load(f)

        vlc_config = unpickle_from(path / "vlc.config")

        return NeuralDbProgressTracker(
            intro_state=IntroState(**args["intro_state"]),
            train_state=TrainState(**args["train_state"]),
            vlc_config=vlc_config,
        )
