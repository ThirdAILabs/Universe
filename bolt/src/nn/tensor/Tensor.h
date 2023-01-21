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
  explicit Tensor(uint32_t dim, std::string name);

  /**
   * Returns the dimension of the vectors in the tensor.
   */
  uint32_t dim() const;

  /**
   * Returns the number of nonzeros in each vector in the tensor. If this is not
   * fixed it will return std::nullopt. If the output is dense then this should
   * be equivalent to calling dim().
   */
  virtual std::optional<uint32_t> numNonzeros(bool use_sparsity) const = 0;

  /**
   * Returns the ith vector in the tensor.
   */
  virtual BoltVector& getVector(uint32_t index) = 0;

  virtual uint32_t batchSize() const = 0;

  virtual ~Tensor() = default;

 private:
  // TODO(Nicholas): Update this to support N dimensions (not required for V0).
  uint32_t _dim;
};

using TensorPtr = std::shared_ptr<Tensor>;

using TensorList = std::vector<TensorPtr>;

}  // namespace thirdai::bolt::nn::tensor