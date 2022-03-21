from typing import List, Callable
from .block_interface import Block
from ..__utils__.builder_vectors import BuilderVector, SparseBuilderVector, DenseBuilderVector
from ..__models__.text_embedding_model_interface import TextEmbeddingModel
import cytoolz as ct

class TextBlock(Block):
  def __init__(self, column: int, embedding_model: TextEmbeddingModel, pipeline: List[Callable]=[]):
    self.column = column
    self.embedding_model = embedding_model
    self.dense = embedding_model.returns_dense_features()
    self.preprocess = lambda str_list: ct.pipe(str_list, *pipeline)

  def process(self, input_row: List[str], shared_feature_vector: BuilderVector = None, idx_offset=0) -> BuilderVector:
    if shared_feature_vector is None:
      idx_offset = 0
      shared_feature_vector = DenseBuilderVector if self.dense else SparseBuilderVector
    preprocessed_list_of_strings = self.preprocess([input_row[self.column]])
    self.embedding_model.embedText(preprocessed_list_of_strings, shared_feature_vector, idx_offset)
    return shared_feature_vector
  
  def feature_dim(self) -> int:
      return self.embedding_model.feature_dim()
    
  def returns_dense_features(self) -> bool:
      return self.embedding_model.returns_dense_features()
