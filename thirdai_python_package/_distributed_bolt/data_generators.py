from thirdai._thirdai import dataset
from typing import Callable
import pandas as pd

class InMemoryDataGenerator:
    def __init__(self, generator_lambda: Callable[[], dataset.BoltDataset]):
        self.generator_lambda = generator_lambda
        self.current_epoch = -1

    def next(self):
        if self.current_epoch == -1:
            self.current_dataset, self.current_labels = self.generator_lambda()

        if not (isinstance(self.current_dataset, list)):
            self.current_dataset = [self.current_dataset]

        self.current_epoch += 1
        return self.current_dataset, self.current_labels

    def get_current_epoch(self):
        return self.current_epoch


class SvmDataGenerator(InMemoryDataGenerator):
    def __init__(self, filename: str, batch_size: int):
        super().__init__(
            lambda: dataset.load_bolt_svm_dataset(
                filename,
                batch_size,
            )
        )