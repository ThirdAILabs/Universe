import numpy as np
from thirdai import dataset

"""
A builder vector is a data structure for composing features
from different blocks into a single vector.
"""
class BuilderVector:
  def addSingleFeature(self, start_dim: int, value: float) -> None:
    return
  
  def addSparseFeatures(self, indices: np.ndarray, values: np.ndarray) -> None:
    return
  
  def addDenseFeatures(self, start_dim: int, values: np.ndarray) -> None:
    return

  def to_bolt_vector(self) -> dataset.BoltVector:
    return

class SparseBuilderVector(BuilderVector):
  def __init__(self) -> None:
    self._indices = []
    self._values = []

  def __str__(self):
    return '[' + ', '.join(f'({idx}, {val})' for idx, val in zip(self._indices, self._values)) + ']'

  def __repr__(self):
    return self.__str__()

  def addSingleFeature(self, start_dim: int, value: float) -> None:
    self._indices.append(start_dim)
    self._values.append(value)

  def addSparseFeatures(self, indices: np.ndarray, values: np.ndarray) -> None:
    self._indices.extend(indices.tolist())
    self._values.extend(values.tolist())
  
  def addDenseFeatures(self, start_dim: int, values: np.ndarray) -> None:
    self._indices.extend(np.arange(start_dim, start_dim + values.shape[0]))
    self._values.extend(values.tolist())
  
  def to_bolt_vector(self) -> None:
    sorted_lists = sorted(zip(self._indices, self._values))
    
    # Deduplicate entries by aggregating the values for the same index.
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

    return dataset.make_sparse_vector(self._indices, self._values)

class DenseBuilderVector(BuilderVector):
  def __init__(self):
    self._values = []

  def __str__(self):
    return '[' + ', '.join([str(val) for val in self._values]) + ']'

  def __repr__(self):
    return self.__str__()

  def addSingleFeature(self, start_dim: int, value: float) -> None:
    self._values.append(value)
  
  def addSparseFeatures(self, indices: np.ndarray, values: np.ndarray) -> None:
    raise Exception
  
  def addDenseFeatures(self, start_dim: int, values: np.ndarray) -> None:
    self._values.extend(values.tolist())
  
  def to_bolt_vector(self) -> dataset.BoltVector:
    return dataset.make_dense_vector(self._values)