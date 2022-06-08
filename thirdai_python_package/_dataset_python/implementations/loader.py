import os
from typing import Tuple
from typing_extensions import Self

from ..interfaces import Source, Parser
from .schema import Schema
from thirdai._thirdai import dataset
from thirdai._thirdai import dataset_internal
import random
import threading


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
        est_num_elems: int = 0,
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
        self._est_num_elems = est_num_elems

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

    def input_dim(self):
        """Returns dimension of input vectors."""
        if self._schema is not None:
            return self._schema.input_dim()
        return 0

    def target_dim(self):
        """Returns dimension of target vectors."""
        if self._schema is not None:
            return self._schema.target_dim()
        return 0

    def __read(self, row_generator, max_rows, destination):
        """Helper function to read up to max_rows rows from
        the source and into the destination batch.
        """
        counter = 0
        for next_row in row_generator:
            destination.append(next_row)
            counter += 1
            if counter == max_rows:
                break

    def __load_all_and_process(self) -> Tuple[dataset.BoltDataset, dataset.BoltDataset]:
        """Helper function to load the whole dataset, processes each sample, and
        generates batches of vector embeddings.
        We want to read lines from file in the main thread and
        process a batch of lines in parallel in other threads.
        Visually, data will be processed like this:

        read batch 0 | read batch 1    | read batch 2    | ...
                     | process batch 0 | process batch 1 | process batch 2 | ...
        time ------------------------------------------------------>

        """

        # Initialization
        file = self._source.open()
        row_generator = self._parser.rows(file)
        processor = dataset_internal.BatchProcessor(
            self._schema._input_blocks,
            self._schema._target_blocks,
            self._batch_size,
            self._est_num_elems,
        )

        # Here, we read the first batch.
        # A large batch size is good for parallelism but a smaller
        # batch size will minimize memory consumption and initial latency.
        # Based on empirical observations, ~65,000 seems to give the best
        # speed.
        old_batch = []
        self.__read(row_generator, 65536, old_batch)

        # Define worker function for feature processing thread.
        def worker(batch):
            processor.process_batch(batch)

        # Read subsequent batches while processing them in parallel.
        while len(old_batch) > 0:
            new_batch = []
            t = threading.Thread(target=worker, daemon=True, args=(old_batch,))

            # Start feature processor thread, read next batch in parallel.
            t.start()
            self.__read(row_generator, 65536, new_batch)

            t.join()
            old_batch = new_batch

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
