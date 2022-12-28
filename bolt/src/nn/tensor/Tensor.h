#pragma once

#include <bolt_vector/src/BoltVector.h>
#include <vector>

namespace thirdai::bolt::nn::ops {

class Op;
using OpPtr = std::shared_ptr<Op>;

}  // namespace thirdai::bolt::nn::ops

namespace thirdai::bolt::nn::tensor {

/**
 * A tensor represents a collection of vectors that are either the inputs to a
 * computation graph or produced by one of its ops.
 */
class Tensor {
 public:
  explicit Tensor(uint32_t dim);

  /**
   * Returns the ith vector in the tensor.
   */
  virtual BoltVector& getVector(uint32_t index) = 0;

  /**
   * Returns the dimension of the vectors in the tensor.
   */
  uint32_t dim() const;

  /**
   * Returns the number of nonzeros in each vector in the tensor. If this is not
   * fixed it will return std::nullopt. If the output is dense then this should
   * be equivalent to calling dim().
   */
  virtual std::optional<uint32_t> numNonzeros() const = 0;

  /**
   * Indicates that the provided op is dependent on this tensor, meaning that it
   * uses it as an input. This is used to construct the computation graph and
   * ensure correct ordering of ops.
   */
  void addDependantOp(ops::OpPtr op);

  /**
   * Returns the dependent ops of the tensor. These are the ops which use the
   * tensor as input.
   */
  const std::vector<ops::OpPtr>& dependantOps() const;

 private:
  // TODO(Nicholas): Update this to support N dimensions (not required for V0).
  uint32_t _dim;

 protected:
  std::vector<ops::OpPtr> _dependant_ops;
};

using TensorPtr = std::shared_ptr<Tensor>;

}  // namespace thirdai::bolt::nn::tensor