import torch
import numpy as np
import tensorflow as tf
import time
import thirdai
from tinygrad.tensor import Tensor


class MatrixBenchmark:

  def __init__(self):
    self.names = []
    self.convert_functions = []
    self.run_functions = []
    self.sum_functions = []
    self.times = []


  def add_method(self, name, run_function, convert_function=lambda a: a, sum_function=lambda a: np.sum(a)):
    self.names.append(name)
    self.run_functions.append(run_function)
    self.convert_functions.append(convert_function)
    self.sum_functions.append(sum_function)
    self.times.append([])

  def run_trial(self, matrix_left_size, matrix_right_size, repeats=1):

    left_matrix = np.random.rand(*matrix_left_size).astype("f")
    right_matrix = np.random.rand(*matrix_right_size).astype("f")

    all_sums = []

    for i, (convert_function, run_function, sum_function) in enumerate(zip(self.convert_functions, self.run_functions, self.sum_functions)):
      
      converted_left = convert_function(left_matrix)
      converted_right = convert_function(right_matrix)

      start = time.time()
      for _ in range(repeats):
        result = run_function(converted_left, converted_right)
        all_sums.append(sum_function(result))
      end = time.time()

      self.times[i].append(end - start)
      

    # Allow max of 0.01% error in the sum of the results
    assert(max(all_sums) / min(all_sums) < 1.0001)

  def print_results(self):
    for name, times in zip(self.names, self.times):
      print(f"Method \"{name}\" took an average time of {sum(times) / len(times)}.")

benchmark = MatrixBenchmark()
benchmark.add_method("Numpy", lambda a, b: a.dot(b))
benchmark.add_method("Pytorch", lambda a, b: a @ b, lambda x: torch.from_numpy(x), lambda x: torch.sum(x).item())
benchmark.add_method("Tensorflow", lambda a, b: a @ b, lambda x: tf.convert_to_tensor(x), lambda x: tf.reduce_sum(x).numpy())
benchmark.add_method("Naive", lambda a, b: thirdai.matrix.naive_matmul(a, b))
benchmark.add_method("Eigen Naive", lambda a, b: thirdai.matrix.naive_eigen_matmul(a, b))
# benchmark.add_method("Eigen Fast, 4 slices", lambda a, b: thirdai.matrix.fast_eigen_matmul(a, b, 4))
# benchmark.add_method("Eigen Fast, 16 slices", lambda a, b: thirdai.matrix.fast_eigen_matmul(a, b, 16))
# benchmark.add_method("Eigen Fast, 64 slices", lambda a, b: thirdai.matrix.fast_eigen_matmul(a, b, 64))
# benchmark.add_method("Eigen Fast, 256 slices", lambda a, b: thirdai.matrix.fast_eigen_matmul(a, b, 256))
# benchmark.add_method("Eigen Fast, 1024 slices", lambda a, b: thirdai.matrix.fast_eigen_matmul(a, b, 1024))
benchmark.add_method("Eigen Fast, 4096 slices", lambda a, b: thirdai.matrix.fast_eigen_matmul(a, b, 4096))
# benchmark.add_method("Eigen Fast, 16384 slices", lambda a, b: thirdai.matrix.fast_eigen_matmul(a, b, 16384))
# benchmark.add_method("Eigen Fast, 65534 slices", lambda a, b: thirdai.matrix.fast_eigen_matmul(a, b, 65534))




repeats = 1
matrix_left_size = (100000, 100)
matrix_right_size = (100, 50)
num_matrices = 10

for _ in range(num_matrices):
  benchmark.run_trial(matrix_left_size, matrix_right_size, repeats)

benchmark.print_results()