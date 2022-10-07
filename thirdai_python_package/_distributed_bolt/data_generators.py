from abc import ABCMeta, abstractmethod
from typing import Callable, List, Optional, Tuple, Union

from thirdai._thirdai import dataset


class TrainGenerator(metaclass=ABCMeta):
    @abstractmethod
    def next(self) -> Tuple[List[dataset.BoltDataset], dataset.BoltDataset]:
        pass

    @abstractmethod
    def get_current_epoch(self) -> int:
        pass


class GenericInMemoryTrainGenerator(TrainGenerator):
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
        self.current_epoch = -1

    def next(self):
        if self.current_epoch == -1:
            self.current_dataset, self.current_labels = self.generator()

        if not (isinstance(self.current_dataset, list)):
            self.current_dataset = [self.current_dataset]

        self.current_epoch += 1
        return self.current_dataset, self.current_labels

    def get_current_epoch(self):
        return self.current_epoch


class SvmTrainGenerator(GenericInMemoryTrainGenerator):
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


class GenericStreamingTrainGenerator(TrainGenerator):
    """
    Wraps a simple dataset generator function into a multi-epoch generator
    ready to pass into the distributed API.
    """

    def __init__(
        self,
        backing_stream: Callable[
            [int], Optional[Tuple[dataset.BoltDataset, dataset.BoltDataset]]
        ],
    ):
        """
        Creates a new GenericStreamingDataGenerator backed by the passed in
        dataset generator function.

        Args:
            backing_stream: A callable function that maps integers representing
            the "id" of the dataset within the stream (e.g. 0 is the first
            dataset) to optional tuples of
            (dataset.BoltDataset, dataset.BoltDataset). This class will manage
            this underlying stream to generate datasets for the model. Inputs
            0 to the max valid dataset should not be None, and everything
             outside that range should be None. The user should expect the
            function to be called over valid inputs (inputs that don't return
            None) multiple times, as each dataset will need to be generated
            once per epoch.
        """
        self.backing_stream = backing_stream
        self.current_epoch = 0
        self.current_dataset_id_within_epoch = 0

    def next(self):
        load = self.backing_stream(self.current_dataset_id_within_epoch)
        if load == None:
            self.current_epoch += 1
            load = self.backing_stream(0)
            self.current_dataset_id_within_epoch = 0

            if load == None:
                raise ValueError("Stream load cannot be None after a restart")

        current_dataset, current_labels = load

        if not (isinstance(current_dataset, list)):
            current_dataset = [current_dataset]

        self.current_dataset_id_within_epoch += 1

        return current_dataset, current_labels

    def get_current_epoch(self):
        return self.current_epoch
