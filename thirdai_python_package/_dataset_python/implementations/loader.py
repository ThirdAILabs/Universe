import os
from typing import Tuple
from typing_extensions import Self

from ..interfaces import Source, Parser
from .schema import Schema
from thirdai._thirdai import dataset
from thirdai._thirdai import dataset_internal
import random


class Loader:
    """A dataset loader and preprocessor.
    This object loads data from a specified source and encodes it as
    vectors according to a specified schema.

    For each sample in the dataset, this loader can produce two types of
    vectors: input vectors and target vectors. Input vectors are passed
    as input into a downstream machine learning model while target vectors
    are what the model learns to predict given the input vectors. If the
    given schema does not define features to be included in target vectors,
    then this loader does not produce target features.

    The source and schema can be set using the set_source() and
    set_schema() methods respectively.
    """

    def __init__(
        self,
        source: Source = None,
        parser: Parser = None,
        schema: Schema = None,
        batch_size: int = 256,
        shuffle: bool = False,
        shuffle_seed: int = random.randint(0, 0xFFFFFFFF),
    ) -> None:
        """Constructor.

        Arguments:
            source: Source object - defines how the dataset is accessed, e.g.
                through a database connector or through the local file system.
            parser: Parser object - defines how individual samples (rows) are retrieved
                from the the data source and parses the sample into a row of features.
            schema: Schema object - identifies the raw features to be processed in each
                sample and how to process them.
            batch_size: int - size of each generated batch of vectors.
            shuffle: bool - whether the dataset's samples are shuffled before being batched.

        Arguments can be omitted in exchange for a builder pattern
        invocation.
        """
        self.set_source(source)
        self.set_parser(parser)
        self.set_schema(schema)
        self.set_batch_size(batch_size)
        self._shuffle_rows = shuffle
        self._shuffle_seed = shuffle_seed

    def set_source(self, source: Source) -> Self:
        """Defines the location of the dataset.

        Arguments:
          location: Source object - defines how the dataset is accessed, e.g.
            through a database connector or through the local file system.
        """
        self._source = source
        return self  ### Returns self so we can chain the set() method calls.

    def set_parser(self, parser: Parser) -> Self:
        """Defines how the dataset can be parsed.

        Arguments:
          format: Parser object - defines how individual samples (rows) are retrieved
            from the the data source and parses the sample into a row of features.
        """
        self._parser = parser
        return self  ### Returns self so we can chain the set() method calls.

    def set_schema(self, schema: Schema) -> Self:
        """Defines the how each sample in the dataset is processed.

        Arguments:
          schema: Schema object - identifies the raw features to be processed in each
            sample and how to process them.
        """
        self._schema = schema
        return self  ### Returns self so we can chain the set() method calls.

    def set_batch_size(self, size: int) -> Self:
        """Sets the batch size.

        Arguments:
          size: int - batch size. Default batch size is 1.
        """
        self._batch_size = size
        return self  ### Returns self so we can chain the set() method calls.

    def shuffle(self, seed: int = None) -> Self:
        """Samples will be shuffled before being batched."""
        self._shuffle_rows = True
        # We use a ternary here instead of setting default seed to random.randint()
        # because for some reason that causes the fault value to be the same every
        # time this function is invoked, instead of getting a new and different
        # random number each time.
        self._shuffle_seed = seed if seed is not None else random.randint(0, 0xFFFFFFFF)
        return self  ### Returns self so we can chain the set() method calls.

    def __load_all_and_process(self) -> Tuple[dataset.BoltDataset, dataset.BoltDataset]:
        """Helper function to load the whole dataset, processes each sample, and
        generates batches of vector embeddings.
        """

        file = self._source.open()
        row_generator = self._parser.rows(file)

        processor = dataset_internal.BatchProcessor(
            self._schema._input_blocks,
            self._schema._target_blocks,
            self._batch_size,
        )
        # Stream rows (samples) and process each one according to the schema.
        counter = 0
        raw_batch = []
        next_row = next(row_generator)
        while next_row is not None:
            raw_batch.append(next_row)
            counter += 1

            # This class is meant to be a parallel streaming data loader.
            # We are slowly transitioning to a C++ implementation and a
            # producer-consumer pattern. However, due to the limitations
            # of the current hybrid implementation, the best we can do
            # now is to read lines from file and process it in batches.
            # A large batch size is good for parallelism but a smaller
            # batch size will minimize memory consumption and latency.
            # Based on empirical observations, 8192 seems to be the
            # sweet spot.
            # TODO: Make this load in true parallel and streaming fashion.
            if counter == 8192:
                processor.process_batch(raw_batch)
                raw_batch = []
                counter = 0

            next_row = next(row_generator)

        if len(raw_batch) > 0:
            processor.process_batch(raw_batch)
            raw_batch = []
            counter = 0

        # Close the source when we are done with it.
        self._source.close()
        # Remember that we have loaded and processed the whole dataset
        # and saved the results in memory.
        return processor.export_in_memory_dataset(
            self._shuffle_rows, self._shuffle_seed
        )

    def get_input_dim(self) -> int:
        """Returns the dimension of input vectors."""
        return self._schema._input_dim

    def get_target_dim(self) -> int:
        """Returns the dimension of target vectors."""
        return self._schema._target_dim

    def processInMemory(self) -> Tuple[dataset.BoltDataset, dataset.BoltDataset]:
        """Produces an in-memory dataset of input and target vectors as specified by
        the schema. The input vectors in the dataset are dense only if all
        input feature blocks return dense features. Input vectors are sparse otherwise.
        The same for target vectors.
        """

        if self._schema is None:
            raise RuntimeError(
                "Dataset: schema is not set. Check that the set_schema() method "
                + "is called before calling process()."
            )

        if len(self._schema._input_blocks) == 0:
            raise RuntimeError(
                "Dataset: schema does not have input blocks. Make sure it is "
                + "constructed with the input_blocks parameter, or that "
                + "the add_input_block() method is called."
            )

        if self._source is None:
            raise RuntimeError(
                "Dataset: source is not set. Check that the set_source() method"
                + " is called before calling process()."
            )

        if self._parser is None:
            raise RuntimeError(
                "Dataset: parser is not set. Check that the set_parser() method"
                + " is called before calling process()."
            )

        return self.__load_all_and_process()
