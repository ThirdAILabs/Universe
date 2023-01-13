from abc import ABC, abstractmethod
from typing import Callable, List, Optional, Tuple, Union

from thirdai import data, dataset
from thirdai.bolt.udt_modifications import _create_data_source


class DistributedDatasetLoader(ABC):
    @abstractmethod
    def next() -> Optional[
        Tuple[
            Union[dataset.BoltDataset, List[dataset.BoltDataset]],
            dataset.BoltDataset,
        ]
    ]:
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
    def load() -> None:
        """
        This function is called only once before the first epoch. As this function is called
        independently inside each worker, it can be used for multiple purposes which includes
        initializing construct for data sources which cannot be pickled across workers(ex. ifstream),
        and if some initialization which needed to done independently for each workers.
        """
        pass


class DistributedUDTDatasetLoader(DistributedDatasetLoader):
    def __init__(
        self,
        train_file: str,
        batch_size: int,
        gcp_credentials_path: str,
        max_in_memory_batches: int,
        data_processor,
    ):
        self.generator = None
        self.data_processor = data_processor
        self.train_file = train_file
        self.batch_size = batch_size
        self.gcp_credentials_path = gcp_credentials_path
        self.max_in_memory_batches = max_in_memory_batches
        self.dataset_finished = False

    def load(self):
        self.generator = self.data_processor.get_dataset_loader(
            _create_data_source(
                self.train_file,
                batch_size=self.batch_size,
                gcp_credentials_path=self.gcp_credentials_path,
            ),
            training=True,
        )

    def next(self):
        if self.dataset_finished:
            return None

        if self.max_in_memory_batches == None:
            load = self.generator.load_in_memory()
            self.dataset_finished = True
        else:
            load = self.generator.load_in_memory(self.max_in_memory_batches)

        return load

    def restart(self):
        self.dataset_finished = False
        self.generator.restart()


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

    def load(self):
        pass

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

    def load(self):
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

        return [x_data], y_data

    def restart(self):
        self.column_map_generator.restart()
