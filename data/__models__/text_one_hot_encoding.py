from typing import List
import mmh3
from .text_embedding_model_interface import TextEmbeddingModel
from ..__utils__.builder_vectors import BuilderVector

class TextOneHotEncoding(TextEmbeddingModel):
  def __init__(self, seed, output_dim):
    self.seed = seed
    self.output_dim = output_dim

  def embedText(self, text: List[str], shared_feature_vector: BuilderVector, offset: int) -> None:
    for string in text:
      hash = mmh3.hash(string)
      shared_feature_vector.addSingleFeature(hash % self.output_dim, 1.0)
  
  def returns_dense_features(self) -> bool:
      return False
    
  def feature_dim(self) -> int:
      return self.output_dim
