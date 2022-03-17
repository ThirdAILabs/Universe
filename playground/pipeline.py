from ast import Call
from typing import Callable, Generator, List, Tuple, Union
from typing_extensions import Self
import cytoolz as ct
from operator import and_
import numpy as np
import random
import mmh3

"""
*** Vector ***
Right now, we have two types of vectors:
1. Sparse vector
2. Dense vector
Soon we may have a third one, the sparsity-agnostic Bolt vector.

Since we want to support all of them, and because different blocks may produce different types of 
vectors, the vectors should share an interface. I propose the following methods:

addSingleFeature(start_dim: int, value: float): Used when the block only produces a single feature.
addSparseFeatures(idxs: np.array[int], vals: np.array[float]): Used when the block produces sparse features.
addDenseFeatures(start_dim: int, vals: np.array[float]): Used when the block produces dense features.

This can look something like the following:
"""
class Vector:
  def addSingleFeature(self, start_dim: int, value: float) -> None:
    return
  
  def addSparseFeatures(self, indices: np.ndarray[np.uint32], values: np.ndarray[np.float32]) -> None:
    return
  
  def addDenseFeatures(self, start_dim: int, values: np.ndarray[np.float32]) -> None:
    return

  def finalize(self):
    return

class SparseVector(Vector):
  def __init__(self) -> None:
    self._indices = []
    self._values = []

  def addSingleFeature(self, start_dim: int, value: float) -> None:
    self._indices.append(start_dim)
    self._values.append(value)

  def addSparseFeatures(self, indices: np.ndarray[np.uint32], values: np.ndarray[np.float32]) -> None:
    self._indices.extend(indices.tolist())
    self._values.extend(values.tolist())
  
  def addDenseFeatures(self, start_dim: int, values: np.ndarray[np.float32]) -> None:
    self._indices.extend(np.arange(start_dim, start_dim + values.shape[0]))
    self._values.extend(values.tolist())
  
  def finalize(self) -> None:
    sorted_lists = sorted(zip(self._indices, self._values))
    # Dedupe
    real_size = -1
    last_idx = -1
    for iv in sorted_lists:
      if iv[0] != last_idx:
        real_size += 1
        last_idx = iv[0]

        self._indices[real_size] = iv[0]
        self._values[real_size] = iv[1]
      
      else:
        self._values[real_size] += iv[1]
    
    real_size += 1

    self._indices = self._indices[:real_size]
    self._values = self._values[:real_size]

class DenseVector(Vector):
  def __init__(self):
    self._values = []

  def addSingleFeature(self, start_dim: int, value: float) -> None:
    self._values.append(value)
  
  def addSparseFeatures(self, indices: np.ndarray[np.uint32], values: np.ndarray[np.float32]) -> None:
    raise Exception
  
  def addDenseFeatures(self, start_dim: int, values: np.ndarray[np.float32]) -> None:
    self._values.extend(values.tolist())
  
  def finalize(self) -> None:
    return

"""
Implementation details:
While Python lists are probably implemented as C++ vectors, they are implemented as vectors of references. 
E.g. A list of integers in python is a vector of references to integers stored in the heap, which makes 
even sequential access very slow in native Python code. When a list is passed to a C++ function through 
PyBind, a new vector is constructed and the values of the list are copied into this new vector. This is 
also slow.

Vectors can be implemented in Python using numpy arrays or in C++ as vectors.
If we implement in Python, then the whole dataset should be stored in Python then wrapped into a dataset 
at the end to prevent copying. Note that we cannot wrap individual vectors or batches in C++ and transfer
control to Python in between because PyBind does not guarantee that a memory location remains valid beyond 
a single C++ function call.
If we implement in C++, we just have to extend the existing vector classes with a few extra methods, wrap 
the methods in a Python interface, and we're fine. And we can easily control data ownership. This seems 
to be the simpler option. 
"""

"""
*** Dataset ***
Also in C++ so we can easily pass to flash and bolt.
"""

"""
I think I just need to start writing them now. 
Let's say... 
"""


class Batch:
  """Input and target vectors can either be dense or sparse
  Assume this directly gives our C++ batch object
  """
  def __init__(self, dense_input: bool, dense_target: bool,
               input_vectors: Union[List[SparseVector], List[DenseVector]]=[], 
               target_vectors: Union[List[SparseVector], List[DenseVector]]=[]):
    self.dense_input = dense_input
    self.dense_target = dense_target
    self.input_vectors = input_vectors
    self.target_vectors = target_vectors
  
  # def add_vector(self, input_vector, target_vector):
  #   self.input_vectors.append(input_vector)
  #   self.target_vectors.append(target_vector)

class SourceLocation:
  """Abstract class for data source location. E.g. S3, local file system, python array
  """
  def open(self):
    """Opens the data file.
    """
    return
  
  def close(self):
    """Closes the data file.
    """
    return

class SourceFormat:
  """Abstract class for data source format. E.g. CSV, or Parquet
  """
  def rows(self, file) -> Generator[List[str],None,None]:
    """Yields the columns of the next row.
    """
    yield


#### CORE COMPONENT 1: BLOCK ####

class Block:
  def __init__(self, column: int, *args):
    return
  
  def process(self, input_row: List[str], 
              shared_feature_vector: Union[DenseVector, SparseVector]=None, 
              idx_offset=0) -> Union[DenseVector, SparseVector]:
    """The block can return either dense or sparse features, depending on 
    what is best for the feature that it produces.
    
    Arguments:
      input_row: a list of columns for a single row.
      shared_feature_vector: a vector that is shared among all blocks operating on 
        a particular row. This make it easier for the pipeline object to 
        concatenate the features produced by each block. If not provided, the 
        method creates its own vector, which could be sparse or dense, whatever 
        is best suited for the feature it produces.
      idx_offset: the offset to shift the feature indices by if the preceeding 
        section of the output vector is occupied by other features, only needed 
        if shared_feature_vector is supplied. 
        Can rename to "start_dim"?.
      
      NOTE: If shared_feature_vector and idx_offset make this method confusing, we 
      can split this into two separate methods, one to be called by the pipeline 
      object and fills out a shared vector, and another to be called by a user 
      and produces its own vector.

    Return value:
      A vector

    """
    return

  def feature_dim():
    """We need to know the dimension of the output vectors.
    This also helps when composing the feature blocks.
    """
    return

  def returns_dense_features(self) -> bool:
    """True if the return value is dense, false otherwise.
    """
    return 



#### CORE COMPONENT 2: SCHEMA ####

class Schema:
  def __init__(self):
    self.blocks = []
    self.offsets = []
  
  def add_vector_blocks(self, blocks: List[Block]) -> None:
    if
    offsets = [0]
    for block in blocks:
      offsets.append(block.feature_dim() + offsets[-1])
    self.blocks.append(blocks)
    self.offsets.append(offsets[:-1])


#### CORE COMPONENT 3: PIPELINE ####

class Pipeline:
  def __init__(self):
    return

  def set_source(self, location: SourceLocation, format: SourceFormat) -> Self:
    """Determine where the data source is located and its format.
    The SourceLocation object defines how to get to the data file.
    The SourceFormat object defines how to get rows from the file and parse 
    them into columns.
    """
    self.source_location = location
    self.source_format = format
    return self ### Returns self so we can chain the set() method calls.
  
  def set_schema(self, schema: Schema) -> Self:
    self.schema = schema
    # Returns vector as dense vector if all input feature blocks are 
    # dense, returns sparse vector otherwise
    feature_vector_is_dense = lambda feature_vector: ct.curried.pipe(
      feature_vector,
      ct.curried.map(lambda block: block.returns_dense_features()),
      ct.curried.accumulate(and_)
    )
    self.returns_dense = ct.curried.map(feature_vector_is_dense, self.schema.blocks)
    
    return self ### Returns self so we can chain the set() method calls.
  
  def process_block(self, tuple, input_row):
    dense, block_set, offset_set = tuple
    # can do conditionals in constructor
    if dense:
      shared_input_vec = DenseVector()
    else:
      shared_input_vec = SparseVector()
    
    for block, offset in zip(block_set, offset_set):
      block.process(input_row, shared_input_vec, offset)
    
    return shared_input_vec

  def process(self, batch_size: int, shuffle: bool=False) -> Generator[Batch, None, None]:
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

    vectors = []

    # For now, we read and process the whole dataset. This makes it much easier to shuffle.
    # Probably switch to a fixed-sized buffer at a later point
    next_row = next(rows)
    while next_row is not None:
      processed_vector = [
        self.process_block(tup, next_row) 
        for tup in zip(self.returns_dense, self.schema.blocks, self.schema.offsets)]
      vectors.append(processed_vector)
      next_row = next(rows)

    if shuffle:
      random.shuffle(vectors)
    
    n_batches = (len(vectors) + batch_size - 1) // batch_size

    for batch in range(n_batches):
      # this can be done in constructor set_schema.
      # Also this is a placeholder for before we have the ideal batch
      if len(self.schema.blocks) == 3:
        batch = ClickThroughBatch()
      elif self.returns_dense[0] ==  
      yield Batch(self.returns_dense_input_vector, self.returns_dense_target_vector, vectors[batch * batch_size : min((batch + 1) * batch_size, len(vectors))])


# Adapt some of my C++ implementations to be compatible with this.
# Or implement some simple ones in Python
  
# Suppose I make a text embedding model that just does one-hot encoding of the
# trigram tokens. THen I need to make one for categorical data as well. Isn't this 
# repetitive? Well, to be fair the one-hot encoding part is the easy part, so most 
# of it is not repeatable.
class TextEmbeddingModel:
  def embedText(self, text: List[str], shared_feature_vector: Union[SparseVector, DenseVector], offset: int) -> None:
    return
  
  def returns_dense_features(self) -> bool:
    return

  def feature_dim(self) -> int:
    """Only needed if used by pipeline object, which is when inputs will
    be processed in batches. This enforces that all output vectors must 
    be the same dimension.
    
    In the case of ColBERT, this enforces that max_tokens must be set if 
    we process in batches.
    """
    return

class TextOneHotEncoding(TextEmbeddingModel):
  def __init__(self, seed, output_dim):
    self.seed = seed
    self.output_dim = output_dim

  def embedText(self, text: List[str], shared_feature_vector: Union[SparseVector, DenseVector], offset: int) -> None:
    for string in text:
      hash = mmh3.hash(string)
      shared_feature_vector.addSingleFeature(hash % self.output_dim, 1.0)
  
  def returns_dense_features(self) -> bool:
      return False
    
  def feature_dim(self) -> int:
      return self.output_dim

class TextBlock(Block):
  def __init__(self, column: int, pipeline: List[Callable], embedding_model: TextEmbeddingModel):
    self.column = column
    self.embedding_model = embedding_model
    self.dense = embedding_model.returns_dense_features()
    self.preprocess = lambda str_list: ct.curried.pipe(str_list, *pipeline)

  def process(self, input_row: List[str], shared_feature_vector: Union[DenseVector, SparseVector] = None, idx_offset=0) -> Union[DenseVector, SparseVector]:
    if shared_feature_vector is None:
      idx_offset = 0
      shared_feature_vector = DenseVector if self.dense else SparseVector
    preprocessed_list_of_strings = self.preprocess(input_row[self.column])
    self.embedding_model.embedText(preprocessed_list_of_strings, shared_feature_vector, idx_offset)
    return shared_feature_vector
  




