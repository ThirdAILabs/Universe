from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Callable, List, Optional, Tuple, Union

from thirdai import bolt, data, dataset
from thirdai.bolt.udt_modifications import _create_data_source

# TODO(Josh/Pratik): Clean up this file and remove the unnecessary DatasetLoaders


class DistributedDatasetLoader(ABC):
    @abstractmethod
    def next() -> Optional[List[dataset.BoltDataset]]:
        """
        This function returns training data and labels if there is training data left for
        ingestion for a epoch else, will return NULL.

        Returns:
            Optional[ Tuple[ Union[dataset.BoltDataset, List[dataset.BoltDataset]], dataset.BoltDataset, ] ]:
                It either returns tuple of training data and training labels or None.
        """
        pass

    @abstractmethod
    def restart() -> None:
        """
        This function is needed to be called before every epoch other than 1st epoch. It moves
        the training data pointer to the front to restart ingestion of training data again.
        """
        pass

    @abstractmethod
    def load(shuffle: bool) -> None:
        """
        This function is called only once before the first epoch. As this function is called
        independently inside each worker, it can be used for multiple purposes which includes
        initializing construct for data sources which cannot be pickled across workers(ex. ifstream),
        and if some initialization which needed to done independently for each workers.
        """
        pass


class DistributedFeaturizerDatasetLoader(DistributedDatasetLoader):
    """
    This is a DistributedDatasetLoader that can accept any featurizer and use it to
    featurize the data.
    """

    def __init__(
        self,
        batch_size,
        data_source_factory,
        max_in_memory_batches=None,
        featurizer=None,
        shuffle=True,
        *args,
        **kwargs,
    ):
        self.featurizer = featurizer
        self.batch_size = batch_size
        self.max_in_memory_batches = max_in_memory_batches
        self.shuffle = shuffle
        self.data_source_factory = data_source_factory
        self.args = args
        self.kwargs = kwargs
        self.dataset_finished = False

    def load(self):
        data_source = self.data_source_factory(*self.args, **self.kwargs)
        self.generator = dataset.DatasetLoader(
            data_source=data_source,
            featurizer=self.featurizer,
            shuffle=self.shuffle,
        )

    def next(self):
        if self.dataset_finished:
            return None

        if self.max_in_memory_batches == None:
            load = self.generator.load_all(batch_size=self.batch_size)
            self.dataset_finished = True
        else:
            load = self.generator.load_some(
                num_batches=self.max_in_memory_batches, batch_size=self.batch_size
            )

        return load

    def restart(self):
        self.generator.restart()


@dataclass
class ValidationContext:
    validation_source: DistributedDatasetLoader
    metrics: List[str]
    sparse_inference: bool
    validation_frequency: int


class DistributedUDTDatasetLoader(DistributedDatasetLoader):
    def __init__(
        self,
        train_file: str,
        batch_size: int,
        data_processor,
        max_in_memory_batches: int = None,
    ):
        self.generator = None
        self.data_processor = data_processor
        self.train_file = train_file
        self.batch_size = batch_size
        self.max_in_memory_batches = max_in_memory_batches
        self.dataset_finished = False

    def load(self, shuffle: bool = True):
        self.generator = self.data_processor.get_dataset_loader(
            _create_data_source(self.train_file),
            training=shuffle,
        )

    def next(self):
        if self.dataset_finished:
            return None

        if self.max_in_memory_batches == None:
            load = self.generator.load_all(batch_size=self.batch_size)
            self.dataset_finished = True
        else:
            load = self.generator.load_some(
                batch_size=self.batch_size, num_batches=self.max_in_memory_batches
            )

        return load

    def restart(self):
        self.dataset_finished = False
        self.generator.restart()


class DistributedColdStartDatasetLoader(DistributedUDTDatasetLoader):
    def __init__(
        self,
        train_file: str,
        batch_size: int,
        max_in_memory_batches: int,
        strong_column_names: List[str],
        weak_column_names: List[str],
        data_processor,
        cold_start_meta_data,
    ):
        self.generator = None
        self.train_file = train_file
        self.strong_column_names = strong_column_names
        self.weak_column_names = weak_column_names
        self.batch_size = batch_size
        self.max_in_memory_batches = max_in_memory_batches
        self.dataset_finished = False
        self.data_processor = data_processor
        self.cold_start_meta_data = cold_start_meta_data

    def load(self, shuffle: bool = True):
        original_data_source = _create_data_source(self.train_file)
        cold_start_data_source = (
            bolt.distributed_preprocessing.preprocess_cold_start_train_source(
                original_data_source,
                self.strong_column_names,
                self.weak_column_names,
                self.data_processor,
                self.cold_start_meta_data,
            )
        )
        self.generator = self.data_processor.get_dataset_loader(
            cold_start_data_source, training=shuffle
        )


class DistributedGenericInMemoryDatasetLoader(DistributedDatasetLoader):
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
        self.dataset_finished = True

    def load(self, shuffle: bool = True):
        pass

    def next(self):
        if self.generated_for_this_epoch:
            return None
        self.generated_for_this_epoch = True

        if self.current_dataset == None:
            self.current_dataset, self.current_labels = self.generator()

            if not (isinstance(self.current_dataset, list)):
                self.current_dataset = [self.current_dataset]

        return self.current_dataset + [self.current_labels]

    def restart(self):
        self.generated_for_this_epoch = False


class DistributedSvmDatasetLoader(DistributedGenericInMemoryDatasetLoader):
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


class DistributedTabularDatasetLoader(DistributedDatasetLoader):
    def __init__(
        self,
        column_map_generator: data.ColumnMapGenerator,
        x_featurizer: data.FeaturizationPipeline,
        y_featurizer: data.FeaturizationPipeline,
        x_cols: List[str],
        y_col: str,
        batch_size: int,
    ):
        self.column_map_generator = column_map_generator
        self.x_featurizer = x_featurizer
        self.y_featurizer = y_featurizer
        self.x_cols = x_cols
        self.y_col = y_col
        self.batch_size = batch_size

    def load(self, shuffle: bool = True):
        pass

    def next(self):
        load = self.column_map_generator.next()
        if load == None:
            return None

        featurized_x = self.x_featurizer.featurize(load)
        featurized_y = self.y_featurizer.featurize(load)

        x_data = featurized_x.convert_to_dataset(
            self.x_cols, batch_size=self.batch_size
        )
        y_data = featurized_y.convert_to_dataset(
            [self.y_col], batch_size=self.batch_size
        )

        # If we only read one batch we return None because the "batch size" of
        # x_data will be less than self.batch_size, which will throw an error
        # when we try to set it in a distributed wrapper. We can remove this
        # when we move to the new dataset class.
        if len(x_data) == 1:
            return None

        return [x_data, y_data]

    def restart(self):
        self.column_map_generator.restart()
