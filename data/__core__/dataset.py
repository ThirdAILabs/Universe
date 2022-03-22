from ..__sources__.source_interfaces import SourceLocation, SourceFormat
from operator import and_
from .schema import Schema
from ..__utils__.builder_vectors import BuilderVector, SparseBuilderVector, DenseBuilderVector
from typing import List, Generator, Tuple
import cytoolz as ct
import random
from thirdai import dataset

class Dataset:
  def __init__(self):
    self._batch_size = 1
    self._shuffle = False
    self._loaded_entire_dataset_in_memory = False

  def set_source(self, location: SourceLocation, format: SourceFormat):
    """Determine where the data source is located and its format.
    The SourceLocation object defines how to get to the data file.
    The SourceFormat object defines how to get rows from the file and parse 
    them into columns.
    """
    self.source_location = location
    self.source_format = format
    return self ### Returns self so we can chain the set() method calls.
  
  def set_schema(self, schema: Schema):
    self.schema = schema
    # Returns input vector as dense vector if all input feature blocks are 
    # dense, returns sparse vector otherwise
    self.returns_dense_input_vector = list(ct.pipe(
      self.schema.input_feature_blocks,
      ct.curried.map(lambda block: block.returns_dense_features()),
      ct.curried.accumulate(and_)
    ))[-1]
    # Returns target vector as dense vector if all target feature blocks are 
    # dense, returns sparse vector otherwise
    # This edge case is super ugly. Fix this later.
    if len(self.schema.target_feature_blocks) == 0:
      self.returns_dense_target_vector = False
    else:  
      self.returns_dense_target_vector = list(ct.pipe(
        self.schema.target_feature_blocks,
        ct.curried.map(lambda block: block.returns_dense_features()),
        ct.curried.accumulate(and_)
      ))[-1]
    return self ### Returns self so we can chain the set() method calls.

  def set_batch_size(self, size: int):
    self._batch_size = size
    return self ### Returns self so we can chain the set() method calls.

  def shuffle(self):
    self._shuffle = True
    return self ### Returns self so we can chain the set() method calls.
  
  def process_row(self, input_row: List[str], dense, blocks, offsets) -> Tuple[BuilderVector, BuilderVector]:

    # Process input vec
    if dense:
      shared_vec = DenseBuilderVector()
    else:
      shared_vec = SparseBuilderVector()

    for block, offset in zip(blocks, offsets):
      block.process(input_row, shared_vec, offset)

    return shared_vec.to_bolt_vector()

  def __load_all_and_process(self):
    if not self._loaded_entire_dataset_in_memory:
      file = self.source_location.open()
      row_generator = self.source_format.rows(file)

      self._input_vectors = []
      self._target_vectors = None if len(self.schema.target_feature_blocks) == 0 else []

      next_row = next(row_generator)
      while next_row is not None:
        input_vec = self.process_row(
          next_row, 
          self.returns_dense_input_vector, 
          self.schema.input_feature_blocks, 
          self.schema.input_feature_offsets)
        
        self._input_vectors.append(input_vec)

        if self._target_vectors:
          target_vec = self.process_row(
            next_row, 
            self.returns_dense_target_vector, 
            self.schema.target_feature_blocks, 
            self.schema.target_feature_offsets)
          self._target_vectors.append(target_vec)

        next_row = next(row_generator)
      
      self.source_location.close()

      self._loaded_entire_dataset_in_memory = True

    if self._shuffle and self._target_vectors:
      temp = list(zip(self._input_vectors, self._target_vectors))
      random.shuffle(temp)
      self._input_vectors, self._target_vectors = zip(*temp)
      # res1 and res2 come out as tuples, and so must be converted to lists.
      self._input_vectors, self._target_vectors = list(self._input_vectors), list(self._target_vectors)
    
    elif self._shuffle:
      random.shuffle(self._input_vectors)
    
    n_batches = (len(self._input_vectors) + self._batch_size - 1) // self._batch_size

    for batch in range(n_batches):
      start_idx = batch * self._batch_size
      end_idx = min((batch + 1) * self._batch_size, len(self._input_vectors))

      yield dataset.BoltInputBatch(
        self._input_vectors[start_idx:end_idx], 
        [] if self._target_vectors is None else self._target_vectors[start_idx:end_idx]
      )

  def __stream_batch_and_process(self):
    file = self.source_location.open()
    row_generator = self.source_format.rows(file)

    next_row = next(row_generator)
    while next_row is not None:
      input_vectors = []
      target_vectors = None if len(self.schema.target_feature_blocks) == 0 else []

      # process a batch
      current_batch_size = 0
      while next_row is not None and current_batch_size < self._batch_size:
        input_vec = self.process_row(
          next_row, 
          self.returns_dense_input_vector, 
          self.schema.input_feature_blocks, 
          self.schema.input_feature_offsets)
        
        input_vectors.append(input_vec)

        if target_vectors:
          target_vec = self.process_row(
            next_row, 
            self.returns_dense_target_vector, 
            self.schema.target_feature_blocks, 
            self.schema.target_feature_offsets)
          target_vectors.append(target_vec)

        next_row = next(row_generator)
        current_batch_size += 1
      
      yield dataset.BoltInputBatch(
        input_vectors, 
        [] if target_vectors is None else target_vectors
      )

    self.source_location.close()

  def process(self) -> Generator[dataset.BoltInputBatch, None, None]:
    """The generator yields a batch of input and target vectors as specified by 
    the schema. The input vectors in the yielded batch are dense only if all 
    input feature blocks return dense features. Input vectors are sparse otherwise.
    The same for target vectors.
    
    This information is readily available before batch generation in 
    self.returns_dense_input_vector and self.returns_dense_target_vector member 
    fields.
    """
    # Loads the whole dataset in memory if we need to shuffle.
    # Otherwise, stream batch by batch.
    # Ultimately, we want to load into a fixed-size buffer. We keep loading the 
    # dataset until the buffer is full, after which we will stream the data batch
    # by batch. This way, we don't have to keep reloading data if the whole dataset
    # fits in memory.
    if self._shuffle:
      return self.__load_all_and_process()
    else:
      return self.__stream_batch_and_process()