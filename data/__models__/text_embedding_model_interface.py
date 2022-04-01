from typing import List
from ..__utils__.builder_vectors import BuilderVector

class TextEmbeddingModel:
  def embedText(self, text: List[str], shared_feature_vector: BuilderVector, offset: int) -> None:
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