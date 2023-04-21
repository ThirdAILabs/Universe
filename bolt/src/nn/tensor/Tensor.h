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
  Tensor(Dims dims, uint32_t nonzeros);

  Tensor(const BoltBatch& batch, uint32_t dim);

  static std::shared_ptr<Tensor> dense(Dims dims);

  static std::shared_ptr<Tensor> sparse(Dims dims, uint32_t nonzeros);

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

  /**
   * Returns the ith vector in the tensor.
   */
  BoltVector& getVector(uint32_t index);

  uint32_t vectorsForSampleStart(uint32_t index_in_batch) const;

  uint32_t vectorsForSampleEnd(uint32_t index_in_batch) const;

  /**
   * Returns the number of vectors in the tensor.
   */
  uint32_t batchSize() const;

  const uint32_t* activeNeuronsPtr() const;

  const float* activationsPtr() const;

  const float* gradientsPtr() const;

 private:
  Dims _dims;
  std::optional<uint32_t> _nonzeros;
  uint32_t _vectors_per_batch_element;

  std::vector<BoltVector> _vectors;

  std::vector<uint32_t> _active_neurons;
  std::vector<float> _activations;
  std::vector<float> _gradients;
};

using TensorPtr = std::shared_ptr<Tensor>;

using TensorList = std::vector<TensorPtr>;

}  // namespace thirdai::bolt::nn::tensor