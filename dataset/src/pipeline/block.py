from builder_vectors import Vector
from typing import List

class Block:
  def __init__(self, column: int, *args):
    return
  
  def process(self, input_row: List[str], 
              shared_feature_vector: Vector=None, 
              idx_offset=0) -> Vector:
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

  def feature_dim(self) -> int:
    """We need to know the dimension of the output vectors.
    This also helps when composing the feature blocks.
    """
    return

  def returns_dense_features(self) -> bool:
    """True if the return value is dense, false otherwise.
    """
    return 
