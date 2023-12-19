from pathlib import Path

from ..utils import assert_file_exists


class TrainState:
    def __init__(
        self,
        learning_rate: float,
        min_epochs: int,
        max_epochs: int,
        freeze_before_train: bool,
        max_in_memory_batches: int,
        current_epoch_number: int,
        is_training_completed: bool,
        **kwargs,
    ):
        self.learning_rate = learning_rate
        self.min_epochs = min_epochs
        self.max_epochs = max_epochs
        self.freeze_before_train = freeze_before_train
        self.max_in_memory_batches = max_in_memory_batches
        self.current_epoch_number = current_epoch_number
        self.is_training_completed = is_training_completed

    def __dict__(self):
        return {
            "learning_rate": self.learning_rate,
            "min_epochs": self.min_epochs,
            "max_epochs": self.max_epochs,
            "freeze_before_train": self.freeze_before_train,
            "max_in_memory_batches": self.max_in_memory_batches,
            "current_epoch_number": self.current_epoch_number,
            "is_training_completed": self.is_training_completed,
        }


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

    def __dict__(self):
        return {
            "num_buckets_to_sample": self.num_buckets_to_sample,
            "fast_approximation": self.fast_approximation,
            "override_number_classes": self.override_number_classes,
            "is_insert_completed": self.is_insert_completed,
        }


class NeuralDbProgressTracker:
    """
    This class will be used to track the current training status of a NeuralDB Mach Model.
    The training state needs to be updated constantly while a model is being trained and
    hence, this should ideally be used inside a callback.

    Given the NeuralDbProgressTracker of the model and the data sources, we should be able to resume the training.
    """

    def __init__(self):
        # These are the introduce state arguments and updated once the introduce document is done
        self._intro_state = IntroState(
            num_buckets_to_sample=None,  # type: ignore
            fast_approximation=None,  # type: ignore
            override_number_classes=None,  # type: ignore
            is_insert_completed=False,
        )

        # These are training arguments and are updated while the training is in progress
        self._train_state = TrainState(
            learning_rate=None,  # type: ignore
            min_epochs=None,  # type: ignore
            max_epochs=None,  # type: ignore
            freeze_before_train=None,  # type: ignore
            max_in_memory_batches=None,  # type: ignore
            current_epoch_number=0,
            is_training_completed=False,
        )
        self.vlc_config = None

    @property
    def num_buckets_to_sample(self):
        return self._intro_state.num_buckets_to_sample

    @num_buckets_to_sample.setter
    def num_buckets_to_sample(self, value):
        self._intro_state.num_buckets_to_sample = value

    @property
    def fast_approximation(self):
        return self._intro_state.fast_approximation

    @fast_approximation.setter
    def fast_approximation(self, value):
        if not isinstance(value, bool):
            raise ValueError("Fast approximation must be a boolean")
        self._intro_state.fast_approximation = value

    @property
    def override_number_classes(self):
        return self._intro_state.override_number_classes

    @override_number_classes.setter
    def override_number_classes(self, value):
        self._intro_state.override_number_classes = value

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
    def learning_rate(self):
        return self._train_state.learning_rate

    @learning_rate.setter
    def learning_rate(self, learning_rate):
        if learning_rate < 0:
            raise ValueError("Negative learning rate not supported")
        self._train_state.learning_rate = learning_rate

    @property
    def min_epochs(self):
        return self._train_state.min_epochs

    @min_epochs.setter
    def min_epochs(self, min_epochs):
        if min_epochs < 0:
            raise ValueError("Negative epochs not supported")
        self._train_state.min_epochs = min_epochs

    @property
    def max_epochs(self):
        return self._train_state.max_epochs

    @max_epochs.setter
    def max_epochs(self, max_epochs):
        if max_epochs < 0:
            raise ValueError("Negative epochs not supported")
        self._train_state.max_epochs = max_epochs

    @property
    def freeze_before_train(self):
        return self._train_state.freeze_before_train

    @freeze_before_train.setter
    def freeze_before_train(self, freeze_before_train):
        self._train_state.freeze_before_train = freeze_before_train

    @property
    def max_in_memory_batches(self):
        return self._train_state.max_in_memory_batches

    @max_in_memory_batches.setter
    def max_in_memory_batches(self, max_in_memory_batches):
        self._train_state.max_in_memory_batches = max_in_memory_batches

    @property
    def current_epoch_number(self):
        return self._train_state.current_epoch_number

    @current_epoch_number.setter
    def current_epoch_number(self, current_epoch_number: int):
        if isinstance(current_epoch_number, int):
            self._train_state.current_epoch_number = current_epoch_number
        else:
            raise TypeError("Can set the property only with an int")

    @property
    def is_training_completed(self):
        return self._train_state.is_training_completed

    @is_training_completed.setter
    def is_training_completed(self, is_training_completed: bool):
        if isinstance(is_training_completed, bool):
            self._train_state.is_training_completed = is_training_completed
        else:
            raise TypeError("Can set the property only with a bool")

    def __dict__(self):
        return {
            "intro_state": self._intro_state.__dict__(),
            "train_state": self._train_state.__dict__(),
        }

    @staticmethod
    def load(arguments_json, vlc_config):
        tracker = NeuralDbProgressTracker()
        tracker._intro_state = IntroState(**(arguments_json["intro_state"]))
        tracker._train_state = TrainState(**(arguments_json["train_state"]))
        tracker.vlc_config = vlc_config
        return tracker
