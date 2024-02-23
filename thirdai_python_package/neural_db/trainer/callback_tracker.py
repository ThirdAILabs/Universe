from pathlib import Path
from typing import List

from thirdai import bolt

from ..utils import pickle_to, unpickle_from


class GetEndLearningRate(bolt.train.callbacks.Callback):
    def __init__(self, base_learning_rate):
        super().__init__()
        self.end_learning_rate = base_learning_rate

    def on_epoch_end(self):
        self.end_learning_rate = super().train_state.learning_rate

    def __getstate__(self):
        return self.__dict__

    def __setstate__(self, state):
        self.__dict__.update(state)


class CallbackTracker:
    def __init__(
        self,
        base_learning_rate: float,
        user_callbacks: List[bolt.train.callbacks.Callback],
    ):
        assert all(
            [
                isinstance(callback, bolt.train.callbacks.Callback)
                for callback in user_callbacks
            ]
        ), "All callback objects should be of type bolt.train.callbacks.Callback"
        self.user_callbacks = user_callbacks
        self.get_lr_callback = GetEndLearningRate(base_learning_rate)

    def user_callbacks_name(self):
        return [
            callback.name()
            if hasattr(callback, "name") and callable(getattr(callback, "name"))
            else "unnamed callback"
            for callback in self.user_callbacks
        ]

    def all_callbacks(self):
        return self.user_callbacks + [self.get_lr_callback]

    def save(self, path: Path):
        pickle_to(self.user_callbacks, path)

    @staticmethod
    def load(path: Path):
        return CallbackTracker(callbacks=unpickle_from(path))

    def get_lr(self):
        return self.get_lr_callback.end_learning_rate
