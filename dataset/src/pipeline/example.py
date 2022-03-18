import imp
from typing import Callable, Generator, List
import cytoolz as ct
import mmh3

from pipeline import Pipeline
from feature_schema import Schema
from text_embedding_model import TextEmbeddingModel
from builder_vectors import Vector, SparseVector, DenseVector
from block import Block
from source import SourceLocation, SourceFormat



class TextOneHotEncoding(TextEmbeddingModel):
  def __init__(self, seed, output_dim):
    self.seed = seed
    self.output_dim = output_dim

  def embedText(self, text: List[str], shared_feature_vector: Vector, offset: int) -> None:
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

  def process(self, input_row: List[str], shared_feature_vector: Vector = None, idx_offset=0) -> Vector:
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