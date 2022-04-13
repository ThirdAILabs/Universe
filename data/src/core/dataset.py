from typing import List, Iterator, Tuple
import random
from .schema import Schema, __BlockList__
from ..sources.source_interface import Source
from ..parsers.parser_interface import Parser
from ..utils.builder_vectors import (
    __BuilderVector__,
    __SparseBuilderVector__,
    __DenseBuilderVector__,
)
from thirdai import dataset


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

    def __init__(self, source: Source=None, parser: Parser=None, schema: Schema=None, batch_size: int=1, shuffle: bool=False, shuffle_seed: int=None):
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
        if shuffle:
            self.shuffle(shuffle_seed)
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

    def shuffle(self, seed: int=None):
        """Samples will be shuffled before being batched."""
        self._shuffle_rows = True
        self._shuffle_seed = seed
        return self  ### Returns self so we can chain the set() method calls.

    def __process_row(
        self, input_row: List[str], blocks: __BlockList__
    ) -> Tuple[__BuilderVector__, __BuilderVector__]:
        """Helper function that processes a single row (sample) into a vector embedding."""

        # Process input vec
        shared_vec = __DenseBuilderVector__() if blocks.is_dense() else __SparseBuilderVector__()

        for block, offset in blocks:
            block.process(input_row, shared_vec, offset)

        return shared_vec.to_bolt_vector()

    def __load_all_and_process(self):
        """Helper function to load the whole dataset, processes each sample, and
        generates batches of vector embeddings.
        """

        # Don't load and process the data all over again if it had been loaded before.
        if not self._loaded_entire_dataset_in_memory:
            file = self._source_location.open()
            row_generator = self._source_format.rows(file)

            self._input_vectors = []
            self._target_vectors = None if len(self._schema.target_blocks) == 0 else []

            # Stream rows (samples) and process each one according to the schema.
            next_row = next(row_generator)
            while next_row is not None:
                input_vec = self.__process_row(
                    next_row,
                    self._schema.input_blocks,
                )

                self._input_vectors.append(input_vec)

                if self._target_vectors:
                    target_vec = self.__process_row(
                        next_row,
                        self._schema.target_blocks,
                    )
                    self._target_vectors.append(target_vec)

                next_row = next(row_generator)

            # Close the source when we are done with it.
            self._source_location.close()
            # Remember that we have loaded and processed the whole dataset
            # and saved the results in memory.
            self._loaded_entire_dataset_in_memory = True

        # Shuffle if necessary.
        if self._shuffle_seed is not None:
            # The random module might be used by other programs, so we have
            # to save this state and revert it when we're done.
            default_random_state = random.getstate()
            if self._last_random_state is not None:
                # we don't want to reseed if we had previously shuffled
                # because then we will have the same shuffle permutation 
                # between epochs.
                random.setstate(self._last_random_state)
            else:
                random.seed(self._shuffle_seed)

        if self._shuffle_rows and self._target_vectors:
            # TODO(Geordie): Zipping seems to be expensive. Look into this if need speedup.
            temp = list(zip(self._input_vectors, self._target_vectors))
            random.shuffle(temp)
            self._input_vectors, self._target_vectors = zip(*temp)
            # input and target come out as tuples, and so must be converted to lists.
            self._input_vectors, self._target_vectors = list(self._input_vectors), list(
                self._target_vectors
            )

        elif self._shuffle_rows:
            random.shuffle(self._input_vectors)
        
        if self._shuffle_seed is not None:
            # Save our random state and revert the random state
            # to what it was before we used the random module.
            self._last_random_state = random.getstate
            random.setstate(default_random_state)

        # Yield the vectors in batches.
        n_batches = (
            len(self._input_vectors) + self._batch_size - 1
        ) // self._batch_size

        for batch in range(n_batches):
            start_idx = batch * self._batch_size
            end_idx = min((batch + 1) * self._batch_size, len(self._input_vectors))

            # TODO(Geordie): Port to C++ soon.
            yield dataset.BoltInputBatch(
                self._input_vectors[start_idx:end_idx],
                []
                if self._target_vectors is None
                else self._target_vectors[start_idx:end_idx],
            )

    def __stream_batch_and_process(self):
        """Helper function to stream samples and process them in batches."""
        file = self._source_location.open()
        row_generator = self._source_format.rows(file)

        next_row = next(row_generator)
        while next_row is not None:
            # New batch, new set of input and target vectors.
            input_vectors = []
            target_vectors = None if len(self._schema.target_blocks) == 0 else []

            # Process a batch according to the schema.
            current_batch_size = 0
            while next_row is not None and current_batch_size < self._batch_size:
                input_vec = self.__process_row(
                    next_row,
                    self._schema.input_blocks,
                )

                input_vectors.append(input_vec)

                if target_vectors:
                    target_vec = self.__process_row(
                        next_row,
                        self._schema.target_blocks,
                    )
                    target_vectors.append(target_vec)

                next_row = next(row_generator)
                current_batch_size += 1

            # Yield the batch. TODO(Geordie): Port to C++ soon.
            yield dataset.BoltInputBatch(
                input_vectors, [] if target_vectors is None else target_vectors
            )

        self._source_location.close()

    def process(self) -> Iterator[dataset.BoltInputBatch]:
        """The generator yields a batch of input and target vectors as specified by
        the schema. The input vectors in the yielded batch are dense only if all
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

        if self._source_location or self._source_format is None:
            raise RuntimeError(
                "Dataset: source is not set. Check that the set_source() method"
                + "is called before calling process()."
            )

        # Loads the whole dataset in memory if we need to shuffle.
        # Otherwise, stream batch by batch.
        # Eventually, whether or not the whole dataset is kept in memory
        # is based on whether it fits, regardless of whether we need to 
        # shuffle.
        if self._shuffle_rows:
            return self.__load_all_and_process()
        else:
            return self.__stream_batch_and_process()
