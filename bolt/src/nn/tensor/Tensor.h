#pragma once

#include <bolt_vector/src/BoltVector.h>
#include <array>
#include <stdexcept>
#include <vector>

namespace thirdai::bolt::nn::tensor {

using Dims = std::vector<uint32_t>;

/**
 * A tensor represents a collection of vectors that are either the inputs to
 * a model or produced by one of its ops.
 */
class Tensor {
 public:
  Tensor(Dims dims, uint32_t nonzeros, bool with_grad);

  Tensor(const uint32_t* indices, const float* values, tensor::Dims dims,
         uint32_t nonzeros, bool with_grad);

  Tensor(const BoltBatch& batch, uint32_t dim);

  static std::shared_ptr<Tensor> dense(Dims dims);

  static std::shared_ptr<Tensor> sparse(Dims dims, uint32_t nonzeros);

  static std::shared_ptr<Tensor> fromArray(const uint32_t* indices,
                                           const float* values,
                                           tensor::Dims dims, uint32_t nonzeros,
                                           bool with_grad);

  static std::shared_ptr<Tensor> convert(const BoltBatch& batch, uint32_t dim);

  static std::shared_ptr<Tensor> convert(const BoltVector& vector,
                                         uint32_t dim);

  /**
   * Returns the dimension of the vectors in the tensor.
   */
  const Dims& dims() const;

  /**
   * Returns the number of nonzeros in each vector in the tensor. If this is not
   * fixed (e.g. for a sparse input tensor) it will return std::nullopt. If the
   * output is dense then this is equivalent to calling dim().
   */
  std::optional<uint32_t> nonzeros() const;

  bool isSparse() const;

  /**
   * Returns the ith vector in the tensor.
   */
  BoltVector& at_2d(uint32_t i) {
    assert(i < _vectors.size());
    return _vectors[i];
  }

  BoltVector& at_3d(uint32_t i, uint32_t j) {
    uint32_t index = i * _dims_3d.at(1) + j;
    assert(index < _vectors.size());
    return _vectors[index];
  }

  const auto& dims2d() const { return _dims_2d; }

  const auto& dims3d() const { return _dims_3d; }

  uint32_t* activeNeuronsAtIndex3d(uint32_t i) {
    assert(index_in_batch < batchSize());
    if (!_nonzeros) {
      throw std::runtime_error("Cannot access sub array of ragged tensor.");
    }
    if (!isSparse()) {
      throw std::runtime_error("Cannot access indices of dense tensor.");
    }
    return _indices.data() + i * _dims_3d.at(1) * (*_nonzeros);
  }

  float* activationsAtIndex3d(uint32_t i) {
    assert(index_in_batch < batchSize());
    if (!_nonzeros) {
      throw std::runtime_error("Cannot access sub array of ragged tensor.");
    }
    return _values.data() + i * _dims_3d.at(1) * (*_nonzeros);
  }

  float* gradientsAtIndex3d(uint32_t i) {
    assert(index_in_batch < batchSize());
    if (!_nonzeros) {
      throw std::runtime_error("Cannot access sub array of ragged tensor.");
    }
    return _gradients.data() + i * _dims_3d.at(1) * (*_nonzeros);
  }
  /**
   * Returns the number of vectors in the tensor.
   */
  uint32_t batchSize() const;

  const uint32_t* indicesPtr() const;
  const float* valuesPtr() const;
  const float* gradientsPtr() const;

  const auto& indices() const { return _indices; }
  auto& indices() { return _indices; }

  const auto& values() const { return _values; }
  auto& values() { return _values; }

  const auto& gradients() const { return _gradients; }
  auto& gradients() { return _gradients; }

 private:
  Dims _dims;
  std::optional<uint32_t> _nonzeros;

  std::array<uint32_t, 2> _dims_2d;

  std::array<uint32_t, 3> _dims_3d;

  std::vector<BoltVector> _vectors;

  std::vector<uint32_t> _indices;
  std::vector<float> _values;
  std::vector<float> _gradients;
};

using TensorPtr = std::shared_ptr<Tensor>;

using TensorList = std::vector<TensorPtr>;

inline bool areDimsEq(const Dims& a, const Dims& b,
                      bool include_last_dim = true) {
  if (a.size() != b.size()) {
    return false;
  }

  uint32_t end = include_last_dim ? a.size() : a.size() - 1;
  for (uint32_t i = 0; i < end; i++) {
    if (a.at(i) != b.at(i)) {
      return false;
    }
  }

  return true;
}

inline std::string toString(const Dims& dims) {
  std::string str = "(";
  for (uint32_t d : dims) {
    str += std::to_string(d) + ", ";
  }

  str.pop_back();  // Remove last ', '
  str.pop_back();

  return str + ")";
}

}  // namespace thirdai::bolt::nn::tensor