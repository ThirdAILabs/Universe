#pragma once

#include <bolt_vector/src/BoltVector.h>
#include <vector>

namespace thirdai::bolt {

/**
 * A tensor represents a collection of vectors that are either the inputs to a
 * model or produced by one of its ops.
 */
class Tensor {
 public:
  Tensor(size_t batch_size, size_t dim, size_t nonzeros, bool with_grad = true);

  Tensor(std::vector<uint32_t>&& indices, std::vector<float>&& values,
         std::vector<size_t>&& lens, size_t dim);

  Tensor(const uint32_t* indices, const float* values, size_t batch_size,
         size_t dim, size_t nonzeros, bool with_grad);

  Tensor(BoltBatch&& batch, size_t dim);

  Tensor(const BoltBatch& batch, size_t dim);

  static std::shared_ptr<Tensor> dense(size_t batch_size, size_t dim);

  static std::shared_ptr<Tensor> sparse(size_t batch_size, size_t dim,
                                        size_t nonzeros);

  static std::shared_ptr<Tensor> sparse(std::vector<uint32_t>&& indices,
                                        std::vector<float>&& values,
                                        std::vector<size_t>&& lens, size_t dim);

  static std::shared_ptr<Tensor> fromArray(const uint32_t* indices,
                                           const float* values,
                                           size_t batch_size, size_t dim,
                                           size_t nonzeros, bool with_grad);

  static std::shared_ptr<Tensor> copy(const BoltBatch& batch, size_t dim);

  static std::shared_ptr<Tensor> convert(BoltBatch&& batch, size_t dim);

  static std::shared_ptr<Tensor> convert(BoltVector&& vector, size_t dim);

  static std::shared_ptr<Tensor> convert(std::vector<BoltVector>&& batch,
                                         size_t dim) {
    BoltBatch bolt_batch(std::move(batch));
    return std::make_shared<Tensor>(std::move(bolt_batch), dim);
  }

  /**
   * Returns the dimension of the vectors in the tensor.
   */
  size_t dim() const;

  /**
   * Returns the number of nonzeros in each vector in the tensor. If this is not
   * fixed (e.g. for a sparse input tensor) it will return std::nullopt. If the
   * output is dense then this is equivalent to calling dim().
   */
  std::optional<size_t> nonzeros() const;

  /**
   * Returns the ith vector in the tensor.
   */
  BoltVector& getVector(size_t index);

  /**
   * Returns if the tensor is sparse.
   */
  bool isSparse() const { return !_vectors.front().isDense(); }

  /**
   * Returns the number of vectors in the tensor.
   */
  size_t batchSize() const;

  const uint32_t* activeNeuronsPtr() const;

  const float* activationsPtr() const;

  std::pair<std::vector<uint32_t>, std::vector<float> > topKIndexValuePair(
      size_t topk);

  const float* gradientsPtr() const;

 private:
  static void checkBatchContents(const BoltBatch& batch, size_t dim);

  // TODO(Nicholas): Update this to support N dimensions (not required for V0).
  size_t _dim;
  std::optional<size_t> _nonzeros;

  std::vector<BoltVector> _vectors;

  std::vector<uint32_t> _active_neurons;
  std::vector<float> _activations;
  std::vector<float> _gradients;
};

using TensorPtr = std::shared_ptr<Tensor>;

using TensorList = std::vector<TensorPtr>;

}  // namespace thirdai::bolt