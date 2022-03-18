from typing import List
from builder_vectors import Vector

class Batch:
  """Input and target vectors can either be dense or sparse
  Assume this directly gives our C++ batch object
  """
  def __init__(self, dense_input: bool, dense_target: bool,
               input_vectors: List[Vector]=[], 
               target_vectors: List[Vector]=None):
    self.dense_input = dense_input
    self.dense_target = dense_target
    self.input_vectors = input_vectors
    self.target_vectors = target_vectors

  def __str__(self):
    printout = "========================================================================\n"
    printout += f"Batch | size = {len(self.input_vectors)}\n\n"
    if self.target_vectors:
      for i, (input_vec, target_vec) in enumerate(zip(self.input_vectors, self.target_vectors)):
        printout += f"Vector {i}:\n"
        printout += f"Input: {input_vec}\n"
        printout += f"Target: {target_vec}\n\n"
    else:
      for i, input_vec in enumerate(self.input_vectors):
        printout += f"Vector {i}: {input_vec}\n\n"
    printout += "========================================================================"
    return printout

  def __repr__(self):
    return self.__str__()
