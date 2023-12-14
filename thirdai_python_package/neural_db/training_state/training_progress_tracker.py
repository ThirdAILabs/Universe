import json
from pathlib import Path
from ..utils import assert_file_exists


class TrainingConfig:
    def __init__(
        self,
        learning_rate: float,
        min_epochs: int,
        max_epochs: int,
        freeze_before_train: bool,
        max_in_memory_batches: int,
        current_epoch_number: int,
        is_training_completed: bool,
    ):
        self.learning_rate = learning_rate
        self.min_epochs = min_epochs
        self.max_epochs = max_epochs
        self.freeze_before_train = freeze_before_train
        self.max_in_memory_batches = max_in_memory_batches
        self.current_epoch_number = current_epoch_number
        self.is_training_completed = is_training_completed

    def __iter__(self):
        yield "learning_rate", self.learning_rate
        yield "min_epochs", self.min_epochs
        yield "max_epochs", self.max_epochs
        yield "freeze_before_train", self.freeze_before_train
        yield "max_in_memory_batches", self.max_in_memory_batches
        yield "current_epoch_number", self.current_epoch_number
        yield "is_training_completed", self.is_training_completed

    def args(self):
        return {
            "learning_rate": self.learning_rate,
            "min_epochs": self.min_epochs,
            "max_epochs": self.max_epochs,
            "freeze_before_train": self.freeze_before_train,
            "max_in_memory_batches": self.max_in_memory_batches,
        }

    def state(self):
        return {
            "current_epoch_number": self.current_epoch_number,
            "is_training_completed": self.is_training_completed,
        }

    @staticmethod
    def load(training_args_path: Path, training_state_path: Path):
        assert_file_exists(training_args_path)
        assert_file_exists(training_state_path)

        with open(training_args_path, "r") as f:
            args = json.load(f)
        with open(training_state_path, "r") as f:
            state = json.load(f)
        return TrainingConfig(
            learning_rate=args["learning_rate"],
            min_epochs=args["min_epochs"],
            max_epochs=args["max_epochs"],
            freeze_before_train=args["freeze_before_train"],
            max_in_memory_batches=args["max_in_memory_batches"],
            current_epoch_number=state["current_epoch_number"],
            is_training_completed=state["is_training_completed"],
        )


class IntroConfig:
    def __init__(
        self,
        num_buckets_to_sample: int,
        fast_approximation: bool,
        override_number_classes: bool,
        is_insert_completed: bool,
    ):
        self.num_buckets_to_sample = num_buckets_to_sample
        self.fast_approximation = fast_approximation
        self.override_number_classes = override_number_classes
        self.is_insert_completed = is_insert_completed

    def args(self):
        return {
            "num_buckets_to_sample": self.num_buckets_to_sample,
            "fast_approximation": self.fast_approximation,
            "override_number_classes": self.override_number_classes,
            "is_insert_completed": self.is_insert_completed,
        }

    @staticmethod
    def load(intro_args_path: Path):
        assert_file_exists(intro_args_path)
        with open(intro_args_path, "r") as f:
            args = json.load(f)
        return IntroConfig(
            num_buckets_to_sample=args["num_buckets_to_sample"],
            fast_approximation=args["fast_approximation"],
            override_number_classes=args["override_number_classes"],
            is_insert_completed=args["is_insert_completed"],
        )


class DataSourceCheckpointConfig:
    def __init__(self, checkpoint_dir: Path, type: str):
        self.source_checkpoint_location = checkpoint_dir / f"{type}_source.csv"
        self.source_arguments_location = checkpoint_dir / f"{type}_arguments.json"

    def args(self):
        return {
            "source_checkpoint_location": str(self.source_checkpoint_location),
            "source_arguments_location": str(self.source_arguments_location),
        }

    def assert_data_source_exists(self):
        assert_file_exists(self.source_checkpoint_location)
        assert_file_exists(self.source_arguments_location)


class CheckpointConfig:
    def __init__(self, checkpoint_dir: Path):
        # Checkpoint dir here refers to model specific directory
        self.neuraldb_model_checkpoint_location = checkpoint_dir / "model.udt"
        self.intro_datasource_config = DataSourceCheckpointConfig(
            checkpoint_dir=checkpoint_dir, type="intro"
        )
        self.train_datasource_config = DataSourceCheckpointConfig(
            checkpoint_dir=checkpoint_dir, type="train"
        )

        self.save_arguments_checkpoint_location = checkpoint_dir / "save_arguments.json"
        self.intro_args_checkpoint_location = checkpoint_dir / "intro_args.json"
        self.train_args_checkpoint_location = checkpoint_dir / "training_args.json"
        self.train_state_checkpoint_location = checkpoint_dir / "training_state.json"

    def args(self):
        return {
            "neuraldb_model_checkpoint_location": str(
                self.neuraldb_model_checkpoint_location
            ),
            "intro_source_args": self.intro_datasource_config.args(),
            "train_source_args": self.train_datasource_config.args(),
            "save_arguments_checkpoint_location": str(
                self.save_arguments_checkpoint_location
            ),
            "intro_args_checkpoint_location": str(self.intro_args_checkpoint_location),
            "train_args_checkpoint_location": str(self.train_args_checkpoint_location),
            "train_state_checkpoint_location": str(
                self.train_state_checkpoint_location
            ),
        }

    def assert_checkpoint_source_exists(self):
        self.intro_datasource_config.assert_data_source_exists()
        self.train_datasource_config.assert_data_source_exists()
        assert_file_exists(self.neuraldb_model_checkpoint_location)


class NeuralDbProgressTracker:
    """
    This class will be used to track the current training status of a NeuralDB Mach Model.
    The training state needs to be updated constantly while a model is being trained and
    hence, this should ideally be used inside a callback.

    Given the TrainingState of the model, we should be able to resume the training.
    """

    def __init__(self):
        # These are the introduce state arguments and updated once the introduce document is done
        self._intro_config = IntroConfig(
            num_buckets_to_sample=None,
            fast_approximation=None,
            override_number_classes=None,
            is_insert_completed=False,
        )

        # These are training arguments and are updated while the training is in progress
        self._train_config = TrainingConfig(
            learning_rate=None,
            min_epochs=None,
            max_epochs=None,
            freeze_before_train=None,
            max_in_memory_batches=None,
            current_epoch_number=0,
            is_training_completed=False,
        )

    @property
    def num_buckets_to_sample(self):
        return self._intro_config.num_buckets_to_sample

    @num_buckets_to_sample.setter
    def num_buckets_to_sample(self, value):
        self._intro_config.num_buckets_to_sample = value

    @property
    def fast_approximation(self):
        return self._intro_config.fast_approximation

    @fast_approximation.setter
    def fast_approximation(self, value):
        if not isinstance(value, bool):
            raise ValueError("Fast approximation must be a boolean")
        self._intro_config.fast_approximation = value

    @property
    def override_number_classes(self):
        return self._intro_config.override_number_classes

    @override_number_classes.setter
    def override_number_classes(self, value):
        self._intro_config.override_number_classes = value

    @property
    def is_insert_completed(self):
        return self._intro_config.is_insert_completed

    @is_insert_completed.setter
    def is_insert_completed(self, is_insert_completed: bool):
        if isinstance(is_insert_completed, bool):
            self._intro_config.is_insert_completed = is_insert_completed
        else:
            raise Exception("Can set the property only with a bool")

    @property
    def learning_rate(self):
        return self._train_config.learning_rate

    @learning_rate.setter
    def learning_rate(self, learning_rate):
        if learning_rate < 0:
            raise ValueError("Negative learning rate not supported")
        self._train_config.learning_rate = learning_rate

    @property
    def min_epochs(self):
        return self._train_config.min_epochs

    @min_epochs.setter
    def min_epochs(self, min_epochs):
        if min_epochs < 0:
            raise ValueError("Negative epochs not supported")
        self._train_config.min_epochs = min_epochs

    @property
    def max_epochs(self):
        return self._train_config.max_epochs

    @max_epochs.setter
    def max_epochs(self, max_epochs):
        if max_epochs < 0:
            raise ValueError("Negative epochs not supported")
        self._train_config.max_epochs = max_epochs

    @property
    def freeze_before_train(self):
        return self._train_config.freeze_before_train

    @freeze_before_train.setter
    def freeze_before_train(self, freeze_before_train):
        self._train_config.freeze_before_train = freeze_before_train

    @property
    def max_in_memory_batches(self):
        return self._train_config.max_in_memory_batches

    @max_in_memory_batches.setter
    def max_in_memory_batches(self, max_in_memory_batches):
        self._train_config.max_in_memory_batches = max_in_memory_batches

    @property
    def current_epoch_number(self):
        return self._train_config.current_epoch_number

    @current_epoch_number.setter
    def current_epoch_number(self, current_epoch_number: bool):
        if isinstance(current_epoch_number, int):
            self._train_config.current_epoch_number = current_epoch_number
        else:
            raise Exception("Can set the property only with an int")

    @property
    def is_training_completed(self):
        return self._train_config.is_training_completed

    @is_training_completed.setter
    def is_training_completed(self, is_training_completed: bool):
        if isinstance(is_training_completed, bool):
            self._train_config.is_training_completed = is_training_completed
        else:
            raise Exception("Can set the property only with a bool")

    def introduce_arguments(self):
        return self._intro_config.args()

    def training_arguments(self):
        return self._train_config.args()

    def training_state(self):
        return self._train_config.state()
