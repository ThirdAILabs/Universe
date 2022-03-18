import numpy as np

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
  
  def addSparseFeatures(self, indices: np.ndarray, values: np.ndarray) -> None:
    return
  
  def addDenseFeatures(self, start_dim: int, values: np.ndarray) -> None:
    return

  def finalize(self):
    return

class SparseVector(Vector):
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