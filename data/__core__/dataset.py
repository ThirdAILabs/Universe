from typing import List, Iterator, Tuple
import random
from .schema import Schema
from ..__blocks__.block_interface import Block
from ..__sources__.source_interfaces import SourceLocation, SourceFormat
from ..__utils__.builder_vectors import BuilderVector, SparseBuilderVector, DenseBuilderVector
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
  
  Example usage:
    dataset = (
      Dataset()
        .set_source(location, format)
        .set_schema(schema)
        .set_batch_size(256)
        .shuffle()
    )

    for batch in dataset.process():
      do_something_to(batch)
  
  Usage notes:
  - The default batch size is 1.
  - If the shuffle() method is called, the whole dataset is loaded into memory.
    Otherwise, rows is streamed from the file in batches.
  
  """
  def __init__(self):
    """Constructor. 
    No argument since it uses a builder pattern;
    Example usage:
    
    dataset = (
      Dataset()
        .set_source(location, format)
        .set_schema(schema)
        .set_batch_size(256)
        .shuffle()
    )

    """
    self._batch_size = 1
    self._shuffle_rows = False
    self._loaded_entire_dataset_in_memory = False

  def set_source(self, location: SourceLocation, format: SourceFormat):
    """Defines the location and format of the dataset.
    
    Arguments:
      location: SourceLocation object - defines how the dataset is accessed, e.g.
        through a database connector or through the local file system.
      format: SourceFormat object - defines how individual samples (rows) are retrieved 
        from the the data source and parses the sample into a row of features.
    """
    self.source_location = location
    self.source_format = format
    return self ### Returns self so we can chain the set() method calls.
  
  def set_schema(self, schema: Schema):
    """Defines the how each sample in the dataset is processed.

    Arguments:
      schema: Schema object - identifies the raw features to be processed in each 
        sample and how to process them.
    """
    self.schema = schema
    # Returns input vector as dense vector if all input feature blocks are 
    # dense, returns sparse vector otherwise
    self.dense_input = all([block.is_dense() for block in self.schema.input_blocks])
    # Returns target vector as dense vector if all target feature blocks are 
    # dense, returns sparse vector otherwise
    # This edge case is super ugly. Fix this later.
    if len(self.schema.target_blocks) == 0:
      self.dense_target = False
    else:  
      self.dense_target = all([block.is_dense() for block in self.schema.target_blocks])
    return self ### Returns self so we can chain the set() method calls.

  def set_batch_size(self, size: int):
    """Sets the batch size.
    
    Arguments:
      size: int - batch size. Default batch size is 1.
    """
    self._batch_size = size
    return self ### Returns self so we can chain the set() method calls.

  def shuffle(self):
    """Samples will be shuffled before being batched.
    """
    self._shuffle_rows = True
    return self ### Returns self so we can chain the set() method calls.
  
  def __process_row(self, input_row: List[str], dense: bool, blocks: List[Block], offsets: List[int]) -> Tuple[BuilderVector, BuilderVector]:
    """Helper function that processes a single row (sample) into a vector embedding.
    """

    # Process input vec
    shared_vec = DenseBuilderVector() if dense else SparseBuilderVector()
    
    for block, offset in zip(blocks, offsets):
      block.process(input_row, shared_vec, offset)

    return shared_vec.to_bolt_vector()

  def __load_all_and_process(self):
    """Helper function to load the whole dataset, processes each sample, and 
    generates batches of vector embeddings.
    """

    # Don't load and process the data all over again if it had been loaded before.
    if not self._loaded_entire_dataset_in_memory:
      file = self.source_location.open()
      row_generator = self.source_format.rows(file)

      self._input_vectors = []
      self._target_vectors = None if len(self.schema.target_blocks) == 0 else []

      # Stream rows (samples) and process each one according to the schema.
      next_row = next(row_generator)
      while next_row is not None:
        input_vec = self.__process_row(
          next_row, 
          self.dense_input, 
          self.schema.input_blocks, 
          self.schema.input_offsets)
        
        self._input_vectors.append(input_vec)

        if self._target_vectors:
          target_vec = self.__process_row(
            next_row, 
            self.dense_target, 
            self.schema.target_blocks, 
            self.schema.target_offsets)
          self._target_vectors.append(target_vec)

        next_row = next(row_generator)

      # Close the source when we are done with it. 
      self.source_location.close()
      # Remember that we have loaded and processed the whole dataset 
      # and saved the results in memory.
      self._loaded_entire_dataset_in_memory = True


    # Shuffle if necessary.
    if self._shuffle_rows and self._target_vectors:
      temp = list(zip(self._input_vectors, self._target_vectors))
      random.shuffle(temp)
      self._input_vectors, self._target_vectors = zip(*temp)
      # input and target come out as tuples, and so must be converted to lists.
      self._input_vectors, self._target_vectors = list(self._input_vectors), list(self._target_vectors)
    
    elif self._shuffle_rows:
      random.shuffle(self._input_vectors)
    
    # Yield the vectors in batches.
    n_batches = (len(self._input_vectors) + self._batch_size - 1) // self._batch_size

    for batch in range(n_batches):
      start_idx = batch * self._batch_size
      end_idx = min((batch + 1) * self._batch_size, len(self._input_vectors))

      yield dataset.BoltInputBatch(
        self._input_vectors[start_idx:end_idx], 
        [] if self._target_vectors is None else self._target_vectors[start_idx:end_idx]
      )

  def __stream_batch_and_process(self):
    """Helper function to stream samples and process them in batches.
    """
    file = self.source_location.open()
    row_generator = self.source_format.rows(file)

    next_row = next(row_generator)
    while next_row is not None:
      # New batch, new set of input and target vectors.
      input_vectors = []
      target_vectors = None if len(self.schema.target_blocks) == 0 else []

      # Process a batch according to the schema.
      current_batch_size = 0
      while next_row is not None and current_batch_size < self._batch_size:
        input_vec = self.__process_row(
          next_row, 
          self.dense_input, 
          self.schema.input_blocks, 
          self.schema.input_offsets)
        
        input_vectors.append(input_vec)

        if target_vectors:
          target_vec = self.__process_row(
            next_row, 
            self.dense_target, 
            self.schema.target_blocks, 
            self.schema.target_offsets)
          target_vectors.append(target_vec)

        next_row = next(row_generator)
        current_batch_size += 1
      
      # Yield the batch.
      yield dataset.BoltInputBatch(
        input_vectors, 
        [] if target_vectors is None else target_vectors
      )

    self.source_location.close()

  def process(self) -> Iterator[dataset.BoltInputBatch, None, None]:
    """The generator yields a batch of input and target vectors as specified by 
    the schema. The input vectors in the yielded batch are dense only if all 
    input feature blocks return dense features. Input vectors are sparse otherwise.
    The same for target vectors.
    
    This information is readily available before batch generation in 
    self.dense_input and self.dense_target member 
    fields.
    """

    if self.schema is None: 
      raise RuntimeError("Dataset: schema is not set. Check that the set_schema() method" + 
                         "is called before calling process().")

    if self.source_location or self.source_format is None: 
      raise RuntimeError("Dataset: source is not set. Check that the set_source() method" + 
                         "is called before calling process().")

    # Loads the whole dataset in memory if we need to shuffle.
    # Otherwise, stream batch by batch.
    # Ultimately, we want to load into a fixed-size buffer. We keep loading the 
    # dataset until the buffer is full, after which we will stream the data batch
    # by batch. This way, we don't have to keep reloading data if the whole dataset
    # fits in memory.
    if self._shuffle_rows:
      return self.__load_all_and_process()
    else:
      return self.__stream_batch_and_process()