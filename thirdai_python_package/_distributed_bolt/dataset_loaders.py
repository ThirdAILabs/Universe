from abc import ABC, abstractmethod
from typing import Callable, List, Optional, Tuple, Union

from thirdai import dataset, deployment, new_dataset


class DatasetLoader(ABC):
    @abstractmethod
    def next() -> Optional[
        Tuple[
            Union[dataset.BoltDataset, List[dataset.BoltDataset]],
            dataset.BoltDataset,
        ]
    ]:
        pass

    @abstractmethod
    def restart() -> None:
        pass


class GenericInMemoryDatasetLoader(DatasetLoader):
    """
    Wraps a generator function that returns a single pair of training and label
    datasets into an in memory data generator ready to pass into the distributed
    API.
    """

    def __init__(
        self,
        generator: Callable[
            [],
            Tuple[
                Union[dataset.BoltDataset, List[dataset.BoltDataset]],
                dataset.BoltDataset,
            ],
        ],
    ):
        self.generator = generator
        self.current_dataset = None
        self.current_labels = None
        self.generated_for_this_epoch = False

    def next(self):
        if self.generated_for_this_epoch:
            return None
        self.generated_for_this_epoch = True

        if self.current_dataset == None:
            self.current_dataset, self.current_labels = self.generator()

            if not (isinstance(self.current_dataset, list)):
                self.current_dataset = [self.current_dataset]

        return self.current_dataset, self.current_labels

    def restart(self):
        self.generated_for_this_epoch = False


class SvmDatasetLoader(GenericInMemoryDatasetLoader):
    """
    Returns a simple in memory data generator ready to pass into the distributed
    API that will read in the given file name with the given batch_size. The
    file name only needs to be present on the target worker, not neccesarily
    this machine.
    """

    def __init__(self, filename: str, batch_size: int):
        super().__init__(
            lambda: dataset.load_bolt_svm_dataset(
                filename,
                batch_size,
            )
        )


class TabularWrapperDatasetLoader(DatasetLoader):
    def __init__(
        self,
        column_map_generator: new_dataset.ColumnMapGenerator,
        featurizer: new_dataset.FeaturizationPipeline,
        columns_in_dataset: List[str],
        batch_size: int,
    ):
        self.column_map_generator = column_map_generator
        self.featurizer = featurizer
        self.columns_in_dataset = columns_in_dataset
        self.batch_size = batch_size

    def next(self):
        load = self.column_map_generator.next()
        if load == None:
            return None

        columns = self.featurizer.featurize(load)

        return columns.convert_to_dataset(
            self.columns_in_dataset, batch_size=self.batch_size
        )

    def restart(self):
        self.column_map_generator.restart()
