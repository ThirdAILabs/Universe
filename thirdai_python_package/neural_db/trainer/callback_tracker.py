from pathlib import Path
from typing import List

from thirdai import bolt

from ..utils import pickle_to, unpickle_from


class CallbackTracker:
    def __init__(self, callbacks: List[bolt.train.callbacks.Callback]):
        assert all(
            [
                isinstance(callback, bolt.train.callbacks.Callback)
                for callback in callbacks
            ]
        ), "All callback objects should be of type bolt.train.callbacks.Callback"
        self.callbacks = callbacks

    def save(self, path: Path):
        pickle_to(self.callbacks, path)

    @staticmethod
    def load(path: Path):
        return CallbackTracker(callbacks=unpickle_from(path))

    def names(self):
        return [
            callback.name if hasattr(callback, "name") else "unnamed callback"
            for callback in self.callbacks
        ]
