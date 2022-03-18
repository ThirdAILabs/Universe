from ast import Call
from typing import Callable, Generator, List, Tuple, Union
import cytoolz as ct
from operator import and_
import numpy as np
import random
import mmh3
from pyparsing import col



#### CORE COMPONENT 3: PIPELINE ####

class Pipeline:
  def __init__(self):
    return

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
  
  def process_row(self, input_row: List[str], dense, blocks, offsets) -> Tuple[Vector, Vector]:

    # Process input vec
    if dense:
      shared_vec = DenseVector()
    else:
      shared_vec = SparseVector()

    for block, offset in zip(blocks, offsets):
      block.process(input_row, shared_vec, offset)
    shared_vec.finalize()

    #############
    # MOCK
    #############

    # shared_vec = shared_vec.to_bolt_vector()

    # TODO: Implement to_bolt_vector() for both sparsevector and dense vector.
    # TODO: Implement BoltVector python bindings
    # btw we cant just build the bolt vector right away because bolt vectors are fixed-size.

    #############

    return shared_vec

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

    if shuffle and target_vectors:
      temp = list(zip(input_vectors, target_vectors))
      random.shuffle(temp)
      input_vectors, target_vectors = zip(*temp)
      # res1 and res2 come out as tuples, and so must be converted to lists.
      input_vectors, target_vectors = list(input_vectors), list(target_vectors)
    
    elif shuffle:
      random.shuffle(input_vectors)
    
    n_batches = (len(input_vectors) + batch_size - 1) // batch_size

    for batch in range(n_batches):
      start_idx = batch * batch_size
      end_idx = min((batch + 1) * batch_size, len(input_vectors))

      #############
      # MOCK UP
      #############

      # TODO: Write BoltInputBatch Python binding.
      # For now, this python binding needs to take vector of input and 
      # label vectors by value instead of l- or r-reference to make it 
      # work with python.

      #############

      yield Batch(
        self.returns_dense_input_vector, 
        self.returns_dense_target_vector, 
        input_vectors[start_idx:end_idx], 
        target_vectors if target_vectors is None else target_vectors[start_idx:end_idx]
        )


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
    self.preprocess = lambda str_list: ct.pipe(str_list, *pipeline)

  def process(self, input_row: List[str], shared_feature_vector: Union[DenseVector, SparseVector] = None, idx_offset=0) -> Union[DenseVector, SparseVector]:
    if shared_feature_vector is None:
      idx_offset = 0
      shared_feature_vector = DenseVector if self.dense else SparseVector
    preprocessed_list_of_strings = self.preprocess([input_row[self.column]])
    self.embedding_model.embedText(preprocessed_list_of_strings, shared_feature_vector, idx_offset)
    return shared_feature_vector
  
  def feature_dim(self) -> int:
      return self.embedding_model.feature_dim()
    
  def returns_dense_features(self) -> bool:
      return self.embedding_model.returns_dense_features()


class PythonObject(SourceLocation):
  def __init__(self, obj) -> None:
    self.obj = obj
  
  def open(self):
    return self.obj

class CsvPythonList(SourceFormat):
  def __init__(self):
    return
  
  def rows(self, file) -> Generator[List[str], None, None]:
    for line in file:
      yield line.split(',')

    # If exhausted
    while True:
      yield None

list_of_passages = [
  "Hello World",
  "I love ThirdAI",
  "i love thirdai so much",
  "Fantastic Bugs and Where to Find them"
]

source_location = PythonObject(list_of_passages)
source_format = CsvPythonList()

pipeline = Pipeline()
pipeline.set_source(source_location, source_format)

map_lower = lambda str_list: [str.lower(s) for s in str_list]
map_word_unigram = lambda str_list: [item for s in str_list for item in s.split(' ')]

text_pipeline = [map_lower, map_word_unigram]
text_embed = TextOneHotEncoding(seed=10, output_dim=1000)
text_block = TextBlock(column=0, pipeline=text_pipeline, embedding_model=text_embed)
schema = Schema(input_feature_blocks=[text_block])

pipeline.set_schema(schema=schema)

for batch in pipeline.process(2, False):
  print(batch)


# TODO: CLEANUP!!!! SPLIT INTO FILESS!!! DOCUMENTTTTT