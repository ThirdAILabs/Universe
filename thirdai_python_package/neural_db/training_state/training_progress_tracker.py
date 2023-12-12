import json
import os


class TrainingProgressTracker:
    """
    This class will be used to track the current training status of a NeuralDB Mach Model.
    The training state needs to be updated constantly while a model is being trained and
    hence, this should ideally be used inside a callback.

    Given the TrainingState of the model, we should be able to resume the training.
    """

    def __init__(self, checkpoint_dir: str, model_id: int):

        # These are the arguments that track where is the model saved
        self._checkpoint_dir = checkpoint_dir
        self._model_id = model_id

        self._save_arguments_checkpoint_location: str = os.path.join(
            checkpoint_dir, str(model_id), "save_arguments.json"
        )
        self._model_checkpoint_location: str = os.path.join(
            checkpoint_dir, str(model_id), f"model.udt"
        )
        self._intro_shard_checkpoint_location: str = os.path.join(
            checkpoint_dir, str(model_id), f"intro_shard.csv"
        )
        self._train_shard_checkpoint_location: str = os.path.join(
            checkpoint_dir, str(model_id), f"train_shard.csv"
        )

        self._intro_args_checkpoint_location: str = os.path.join(
            checkpoint_dir, str(model_id), f"intro_args.json"
        )
        self._train_args_checkpoint_location: str = os.path.join(
            checkpoint_dir, str(model_id), f"training_args.json"
        )
        self._train_state_checkpoint_location: str = os.path.join(
            checkpoint_dir, str(model_id), f"training_state.json"
        )

        # These are the introduce state arguments and updated once the introduce document is done
        self._num_buckets_to_sample: int = None
        self._fast_approximation: bool = None
        self._override_number_classes: int = None
        self._is_insert_completed: bool = False

        # These are training arguments and are updated while the training is in progress
        self._learning_rate: float = None
        self._min_epochs: int = None
        self._max_epochs: int = None
        self._current_epoch_number: int = 0
        self._is_training_completed: bool = False

    @property
    def num_buckets_to_sample(self):
        return self._num_buckets_to_sample

    @num_buckets_to_sample.setter
    def num_buckets_to_sample(self, value):
        self._num_buckets_to_sample = value

    @property
    def fast_approximation(self):
        return self._fast_approximation

    @fast_approximation.setter
    def fast_approximation(self, value):
        if not isinstance(value, bool):
            raise ValueError("Fast approximation must be a boolean")
        self._fast_approximation = value

    @property
    def override_number_classes(self):
        return self._override_number_classes

    @override_number_classes.setter
    def override_number_classes(self, value):
        self._override_number_classes = value

    @property
    def learning_rate(self):
        return self._learning_rate

    @learning_rate.setter
    def learning_rate(self, learning_rate):
        if learning_rate < 0:
            raise ValueError("Negative learning rate not supported")
        self._learning_rate = learning_rate

    @property
    def min_epochs(self):
        return self._min_epochs

    @min_epochs.setter
    def min_epochs(self, min_epochs):
        if min_epochs < 0:
            raise ValueError("Negative epochs not supported")
        self._min_epochs = min_epochs

    @property
    def max_epochs(self):
        return self._max_epochs

    @max_epochs.setter
    def max_epochs(self, max_epochs):
        if max_epochs < 0:
            raise ValueError("Negative epochs not supported")
        self._max_epochs = max_epochs

    @property
    def epochs(self):
        return self._epochs

    @property
    def checkpoint_dir(self):
        return self._checkpoint_dir

    @property
    def model_id(self):
        return self._model_id

    @property
    def model_checkpoint_location(self):
        return self._model_checkpoint_location

    @property
    def save_args_checkpoint_location(self):
        return self._save_arguments_checkpoint_location

    @property
    def intro_args_checkpoint_location(self):
        return self._intro_args_checkpoint_location

    @property
    def train_args_checkpoint_location(self):
        return self._train_args_checkpoint_location

    @property
    def train_state_checkpoint_location(self):
        return self._train_state_checkpoint_location

    @property
    def is_insert_completed(self):
        return self._is_insert_completed

    @is_insert_completed.setter
    def is_insert_completed(self, is_insert_completed: bool):
        if isinstance(is_insert_completed, bool):
            self._is_insert_completed = is_insert_completed
        else:
            raise Exception("Can set the property only with a bool")

    @property
    def is_training_completed(self):
        return self._is_training_completed

    @is_training_completed.setter
    def is_training_completed(self, is_training_completed: bool):
        if isinstance(is_training_completed, bool):
            self._is_training_completed = is_training_completed
        else:
            raise Exception("Can set the property only with a bool")

    @property
    def current_epoch_number(self):
        return self._current_epoch_number

    @current_epoch_number.setter
    def current_epoch_number(self, current_epoch_number: bool):
        if isinstance(current_epoch_number, int):
            self._current_epoch_number = current_epoch_number
        else:
            raise Exception("Can set the property only with an int")

    def save_arguments(self):
        arguments = {
            "checkpoint_dir": self._checkpoint_dir,
            "model_id": self.model_id,
        }
        return json.dumps(arguments, indent=4)

    def introduce_arguments(self):
        arguments = {
            "num_buckets_to_sample": self.num_buckets_to_sample,
            "fast_approximation": self.fast_approximation,
            "override_number_classes": self.override_number_classes,
            "is_insert_completed": self.is_insert_completed,
        }
        return json.dumps(arguments, indent=4)

    def training_arguments(self):
        arguments = {
            "learning_rate": self.learning_rate,
            "min_epochs": self.min_epochs,
            "max_epochs": self.max_epochs,
        }
        return json.dumps(arguments, indent=4)

    def training_state(self):
        state = {
            "is_training_completed": self.is_training_completed,
            "current_epoch_number": self.current_epoch_number,
        }
        return json.dumps(state, indent=4)

    @classmethod
    def load(
        cls,
        save_location: dict,
        introduce_arguments: dict,
        training_arguments: dict,
        training_state: dict,
    ):
        instance = cls(
            checkpoint_dir=save_location["checkpoint_dir"],
            model_id=save_location["model_id"],
        )
        instance.num_buckets_to_sample = introduce_arguments["num_buckets_to_sample"]
        instance.fast_approximation = introduce_arguments["fast_approximation"]
        instance.override_number_classes = introduce_arguments[
            "override_number_classes"
        ]
        instance.is_insert_completed = introduce_arguments["is_insert_completed"]

        instance.learning_rate = training_arguments["learning_rate"]
        instance.min_epochs = training_arguments["min_epochs"]
        instance.max_epochs = training_arguments["max_epochs"]

        instance.current_epoch_number = training_state["current_epoch_number"]
        instance.is_training_completed = training_state["is_training_completed"]
