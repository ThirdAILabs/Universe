from typing import Callable, List, Optional, Tuple, Union

from thirdai._thirdai import dataset

# These classes implement the DatasetGenerator interact, but when I tried making
# them extend it explicitly using Pybind I ran into problems pickling these
# classes, since they didn't know how to pickle the parent class. Since we
# currently don't need to pass DatasetGenerators into C++, we can leave the
# inheritance as python style "duck" inheritance for now (if it quacks like a
# DatasetGenerator then its a DatasetGenerator).


class GenericInMemoryDatasetLoader:
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


class GenericStreamingDatasetLoader:
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
        self.current_dataset_id_within_epoch = 0

    def next(self):
        load = self.backing_stream(self.current_dataset_id_within_epoch)
        if load == None:
            return None

        current_dataset, current_labels = load
        if not (isinstance(current_dataset, list)):
            current_dataset = [current_dataset]

        self.current_dataset_id_within_epoch += 1

        return current_dataset, current_labels

    def restart(self):
        self.current_dataset_id_within_epoch = 0


# This gets around having to write serialization code for all of our
# batch processors
# TODO(Josh): We should probably write all of the serialization code
class DatasetLoaderFactoryWrapper:
    def __init__(
        self,
        data_loader: Union[Tuple[str, int], dataset.DataLoader],
        model_pipeline,
        max_in_memory_batches,
    ):
        self.data_loader = data_loader
        self.dataset_loader_factory = model_pipeline.dataset_loader_factory
        self.max_in_memory_batches = max_in_memory_batches
        self.initialized = False

    def next(self):
        if not self.initialized:
            if isinstance(self.data_loader, tuple):
                self.dataset_loader = (
                    self.dataset_loader_factory.get_labeled_dataset_loader(
                        max_in_memory_batches=self.max_in_memory_batches,
                        training=True,
                        data_loader=dataset.SimpleFileDataLoader(
                            filename=self.data_loader[0],
                            target_batch_size=self.data_loader[1],
                        ),
                    )
                )
            else:
                self.dataset_loader = (
                    self.dataset_loader_factory.get_labeled_dataset_loader(
                        max_in_memory_batches=self.max_in_memory_batches,
                        training=True,
                        data_loader=self.data_loader,
                    )
                )

        self.initialized = True
        return self.dataset_loader.next()

    def restart(self):
        self.dataset_loader.restart()
