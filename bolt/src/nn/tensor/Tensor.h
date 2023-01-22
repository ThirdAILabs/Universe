#pragma once

#include <bolt_vector/src/BoltVector.h>
#include <vector>

namespace thirdai::bolt::nn::tensor {

/**
 * A tensor represents a collection of vectors that are either the inputs to a
 * model or produced by one of its ops.
 */
class Tensor {
 public:
  Tensor(uint32_t batch_size, uint32_t dim, uint32_t nonzeros);

  Tensor(BoltBatch&& batch, uint32_t dim);

  static std::shared_ptr<Tensor> dense(uint32_t batch_size, uint32_t dim);

  static std::shared_ptr<Tensor> sparse(uint32_t batch_size, uint32_t dim,
                                        uint32_t nonzeros);

  static std::shared_ptr<Tensor> convert(BoltBatch&& batch, uint32_t dim);

  /**
   * Returns the dimension of the vectors in the tensor.
   */
  uint32_t dim() const;

  /**
   * Returns the number of nonzeros in each vector in the tensor. If this is not
   * fixed it will return std::nullopt. If the output is dense then this should
   * be equivalent to calling dim().
   */
  std::optional<uint32_t> nonzeros() const;

  /**
   * Returns the ith vector in the tensor.
   */
  BoltVector& getVector(uint32_t index);

  /**
   * Returns the batch size of the data in the tensor.
   */
  uint32_t batchSize() const;

  const uint32_t* activeNeuronsPtr() const;

  const float* activationsPtr() const;

  const float* gradientsPtr() const;

 private:
  // TODO(Nicholas): Update this to support N dimensions (not required for V0).
  uint32_t _dim;
  std::optional<uint32_t> _nonzeros;

  std::vector<BoltVector> _vectors;

  std::vector<uint32_t> _active_neurons;
  std::vector<float> _activations;
  std::vector<float> _gradients;
};

using TensorPtr = std::shared_ptr<Tensor>;

using TensorList = std::vector<TensorPtr>;

}  // namespace thirdai::bolt::nn::tensor