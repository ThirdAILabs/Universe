from typing import List, Iterator, Tuple
import random
from .schema import Schema, __BlockList__
from sources.source_interface import Source
from parsers.parser_interface import Parser
from utils.builder_vectors import (
    __BuilderVector__,
    __SparseBuilderVector__,
    __DenseBuilderVector__,
)
from thirdai import dataset
import time

class Dataset:
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

    Usage notes:
    - The default batch size is 1.
    - If the shuffle() method is called, the whole dataset is loaded into memory.
      Otherwise, rows is streamed from the file in batches.
    - Calling process() before setting source, parser and schema

    """

    def __init__(
        self,
        source: Source = None,
        parser: Parser = None,
        schema: Schema = None,
        batch_size: int = 256,
        shuffle: bool = False,
        shuffle_seed: int = 0,
    ):
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
        self._last_random_state = None
        self._loaded_entire_dataset_in_memory = False

    def set_source(self, source: Source):
        """Defines the location of the dataset.

        Arguments:
          location: Source object - defines how the dataset is accessed, e.g.
            through a database connector or through the local file system.
        """
        self._source = source
        return self  ### Returns self so we can chain the set() method calls.

    def set_parser(self, parser: Parser):
        """Defines how the dataset can be parsed.

        Arguments:
          format: Parser object - defines how individual samples (rows) are retrieved
            from the the data source and parses the sample into a row of features.
        """
        self._parser = parser
        return self  ### Returns self so we can chain the set() method calls.

    def set_schema(self, schema: Schema):
        """Defines the how each sample in the dataset is processed.

        Arguments:
          schema: Schema object - identifies the raw features to be processed in each
            sample and how to process them.
        """
        self._schema = schema
        return self  ### Returns self so we can chain the set() method calls.

    def set_batch_size(self, size: int):
        """Sets the batch size.

        Arguments:
          size: int - batch size. Default batch size is 1.
        """
        self._batch_size = size
        return self  ### Returns self so we can chain the set() method calls.

    def shuffle(self, seed: int = 0):
        """Samples will be shuffled before being batched."""
        self._shuffle_rows = True
        self._shuffle_seed = seed
        return self  ### Returns self so we can chain the set() method calls.

    def __load_all_and_process(self):
        """Helper function to load the whole dataset, processes each sample, and
        generates batches of vector embeddings.
        """

        file = self._source.open()
        row_generator = self._parser.rows(file)

        processor = dataset.BatchProcessor(
            self._schema.input_blocks.blocks,
            self._schema.target_blocks.blocks,
            self._batch_size)
        # Stream rows (samples) and process each one according to the schema.
        counter = 0
        raw_batch = []
        next_row = next(row_generator)
        actual_processing_time = 0
        while next_row is not None:
            raw_batch.append(next_row)
            counter += 1

            if counter == 8192:
                start = time.time()
                processor.process_batch(raw_batch)
                now = time.time()
                actual_processing_time += now - start
                raw_batch = []
                counter = 0
            
            next_row = next(row_generator)

        if len(raw_batch) > 0:
            start = time.time()
            processor.process_batch(raw_batch)
            now = time.time()
            actual_processing_time += now - start
            raw_batch = []
            counter = 0

        print("Actual processing time:", actual_processing_time)
        # Close the source when we are done with it.
        self._source.close()
        # Remember that we have loaded and processed the whole dataset
        # and saved the results in memory.
        return processor.export_in_memory_dataset(shuffle=self._shuffle_rows, shuffle_seed=self._shuffle_seed)

    def processInMemory(self) -> dataset.BoltDataset:
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

        if len(self._schema.input_blocks) == 0:
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
