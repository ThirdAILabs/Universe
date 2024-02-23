from pathlib import Path
from typing import List

from thirdai import bolt

from ..utils import pickle_to, unpickle_from


class GetEndLearningRate(bolt.train.callbacks.Callback):
    def __init__(self):
        super().__init__()

    def on_epoch_end(self):
        self.current_learning_rate = self.train_state.learning_rate

    def __getstate__(self):
        return self.__dict__

    def __setstate__(self, state):
        self.__dict__.update(state)


class CallbackTracker:
    def __init__(
        self,
        user_callbacks: List[bolt.train.callbacks.Callback],
    ):
        assert all(
            [
                isinstance(callback, bolt.train.callbacks.Callback)
                for callback in user_callbacks
            ]
        ), "All callback objects should be of type bolt.train.callbacks.Callback"
        self.user_callbacks = user_callbacks
        self.LR_callback = GetEndLearningRate()

    def user_callbacks_name(self):
        return [
            callback.name()
            if hasattr(callback, "name") and callable(getattr(callback, "name"))
            else "unnamed callback"
            for callback in self.user_callbacks
        ]

    def all_callbacks(self):
        return self.user_callbacks + [self.LR_callback]

    def save(self, path: Path):
        pickle_to(self.user_callbacks, path)

    @staticmethod
    def load(path: Path):
        return CallbackTracker(user_callbacks=unpickle_from(path))

    def get_lr(self):
        return self.LR_callback.current_learning_rate
