from source import SourceLocation, SourceFormat
from operator import and_
from feature_schema import Schema
from placeholder_batch import Batch
from builder_vectors import Vector, SparseVector, DenseVector
from typing import List, Generator, Tuple
import cytoolz as ct
import random
from thirdai import dataset

class Pipeline:
  def __init__(self):
    self._batch_size = 1
    self._shuffle = False

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
  
  def process_row(self, input_row: List[str], dense, blocks, offsets) -> Tuple[Vector, Vector]:

    # Process input vec
    if dense:
      shared_vec = DenseVector()
    else:
      shared_vec = SparseVector()

    for block, offset in zip(blocks, offsets):
      block.process(input_row, shared_vec, offset)


    #############
    # MOCK
    #############

    # shared_vec = shared_vec.to_bolt_vector()

    # TODO: Implement to_bolt_vector() for both sparsevector and dense vector.
    # TODO: Implement BoltVector python bindings
    # btw we cant just build the bolt vector right away because bolt vectors are fixed-size.

    #############

    return shared_vec.to_bolt_vector()

  def process(self) -> Generator[Batch, None, None]:
    """The generator yields a batch of input and target vectors as specified by 
    the schema. The input vectors in the yielded batch are dense only if all 
    input feature blocks return dense features. Input vectors are sparse otherwise.
    The same for target vectors.
    
    This information is readily available before batch generation in 
    self.returns_dense_input_vector and self.returns_dense_target_vector member 
    fields.
    """
    # Iterate through file
    file = self.source_location.open()
    rows = self.source_format.rows(file)

    input_vectors = []
    target_vectors = None if len(self.schema.target_feature_blocks) == 0 else []

    # For now, we read and process the whole dataset. This makes it much easier to shuffle.
    # Probably switch to a fixed-sized buffer at a later point
    next_row = next(rows)
    while next_row is not None:
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

      next_row = next(rows)

    if self._shuffle and target_vectors:
      temp = list(zip(input_vectors, target_vectors))
      random.shuffle(temp)
      input_vectors, target_vectors = zip(*temp)
      # res1 and res2 come out as tuples, and so must be converted to lists.
      input_vectors, target_vectors = list(input_vectors), list(target_vectors)
    
    elif self._shuffle:
      random.shuffle(input_vectors)
    
    n_batches = (len(input_vectors) + self._batch_size - 1) // self._batch_size

    for batch in range(n_batches):
      start_idx = batch * self._batch_size
      end_idx = min((batch + 1) * self._batch_size, len(input_vectors))

      #############
      # MOCK UP
      #############

      # TODO: Write BoltInputBatch Python binding.
      # For now, this python binding needs to take vector of input and 
      # label vectors by value instead of l- or r-reference to make it 
      # work with python.

      #############

      yield dataset.BoltInputBatch(
        input_vectors[start_idx:end_idx], 
        [] if target_vectors is None else target_vectors[start_idx:end_idx]
      )