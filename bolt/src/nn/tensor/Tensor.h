#pragma once

#include <bolt_vector/src/BoltVector.h>
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
  BoltVector& getVector(uint32_t index);

  constexpr uint32_t innerDim3d() const { return _inner_dim_3d; }

  constexpr uint32_t rangeStart(uint32_t index_in_batch) const {
    return index_in_batch * innerDim3d();
  }

  constexpr uint32_t rangeEnd(uint32_t index_in_batch) const {
    return (index_in_batch + 1) * innerDim3d();
  }

  float* activationsAtIndex3d(uint32_t index_in_batch) {
    assert(index_in_batch < batchSize());
    return _activations.data() + index_in_batch * innerDim3d() * _dims.back();
  }

  float* gradientsAtIndex3d(uint32_t index_in_batch) {
    assert(index_in_batch < batchSize());
    return _gradients.data() + index_in_batch * innerDim3d() * _dims.back();
  }
  /**
   * Returns the number of vectors in the tensor.
   */
  uint32_t batchSize() const;

  const uint32_t* activeNeuronsPtr() const;

  const float* activationsPtr() const;

  const float* gradientsPtr() const;

  const auto& activeNeurons() const { return _active_neurons; }

  auto& activeNeurons() { return _active_neurons; }

  const auto& activations() const { return _activations; }

  auto& activations() { return _activations; }

  const auto& gradients() const { return _gradients; }

  auto& gradients() { return _gradients; }

 private:
  Dims _dims;
  std::optional<uint32_t> _nonzeros;
  uint32_t _inner_dim_3d;

  std::vector<BoltVector> _vectors;

  std::vector<uint32_t> _active_neurons;
  std::vector<float> _activations;
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