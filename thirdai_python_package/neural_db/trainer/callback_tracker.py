from pathlib import Path
from typing import List

from thirdai import bolt

from ..utils import pickle_to, unpickle_from

class GetEndLearningRate(bolt.train.callbacks.Callback):
    def __init__(self):
        self.current_learning_rate = 0
    
    def on_epoch_end(self):
        self.end_learning_rate = super().train_state.learning_rate


class CallbackTracker:
    def __init__(self, user_callbacks: List[bolt.train.callbacks.Callback]):
        assert all(
            [
                isinstance(callback, bolt.train.callbacks.Callback)
                for callback in user_callbacks
            ]
        ), "All callback objects should be of type bolt.train.callbacks.Callback"
        self.user_callbacks = user_callbacks
        self.get_lr_callback = GetEndLearningRate()

    def user_callbacks_name(self):
        return [
            callback.name() if hasattr(callback, "name") and callable(getattr(callback, 'name')) else "unnamed callback"
            for callback in self.user_callbacks
        ]
    
    def all_callbacks(self):
        return [self.get_lr_callback] + [self.user_callbacks]
    
    def save(self, path: Path):
        pickle_to(self.user_callbacks, path)

    @staticmethod
    def load(path: Path):
        return CallbackTracker(callbacks=unpickle_from(path))
    
    def get_lr(self):
        return self.get_lr_callback.end_learning_rate

    
