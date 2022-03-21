from typing import List
from .text_embedding_model_interface import TextEmbeddingModel
from ..__utils__.builder_vectors import BuilderVector

class CustomDenseTextEmbedding(TextEmbeddingModel):
  def __init__(self, embed_fn, out_dim):
    self.embed_fn = embed_fn
    self.out_dim = out_dim
  
  def feature_dim(self) -> int:
    return self.out_dim

  def embedText(self, text: List[str], shared_feature_vector: BuilderVector, offset: int) -> None:
    shared_feature_vector.addDenseFeatures(offset, self.embed_fn(text))
