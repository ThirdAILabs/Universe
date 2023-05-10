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
   * Returns the dimensions of the tensor.
   */
  const Dims& dims() const { return _dims; }

  /**
   * Returns the dimensions of the tensor if the tensor is reshaped as 2D while
   * preserving the last dimension. For example given a tensor with dimensions
   * (2, 3, 4) this method would return (6, 4)
   */
  const auto& dims2d() const { return _dims_2d; }

  /**
   * Returns the dimensions of the tensor if the tensor is reshaped as 3D while
   * preserving the first and last dimensions. For example given a tensor with
   * dimensions (2, 3, 4, 5) this method would return (2, 12, 5)
   */
  const auto& dims3d() const { return _dims_3d; }

  /**
   * Returns the number of nonzeros in each vector in the tensor. If this is not
   * fixed (e.g. for a sparse input tensor) it will return std::nullopt. If the
   * output is dense then this is equivalent to calling dim().
   */
  std::optional<uint32_t> nonzeros() const { return _nonzeros; }

  /**
   * Returns if the tensor is sparse.
   */
  bool isSparse() const { return !_vectors.front().isDense(); }

  /**
   * Accesses the i-th vector of the tensor, treating the tensor as if it is 2d.
   */
  BoltVector& at_2d(uint32_t i);

  /**
   * Accesses the (i,j)-th vector of the tensor, treating the tensor as if it is
   * 3d.
   */
  BoltVector& at_3d(uint32_t i, uint32_t j);

  /**
   * Treats the tensor as 3d. Performs indexing on the outer dimension and
   * returns a pointer to the indices. Throws if the tensor is dense or if the
   * indices are not continuously activated.
   */
  uint32_t* indicesAtIndex3d(uint32_t i);

  /**
   * Treats the tensor as 3d. Performs indexing on the outer dimension and
   * returns a pointer to the indices. Throws if the tensor if the values are
   * not continuously activated.
   */
  float* valuesAtIndex3d(uint32_t i);

  /**
   * Treats the tensor as 3d. Performs indexing on the outer dimension and
   * returns a pointer to the gradients. Throws if the tensor does not have
   * gradients or if the gradients are not continuously activated.
   */
  float* gradientsAtIndex3d(uint32_t i);

  /**
   * Returns the number of vectors in the tensor.
   */
  uint32_t batchSize() const { return _dims.front(); }

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