#pragma once

#include <bolt_vector/src/BoltVector.h>
#include <vector>

namespace thirdai::bolt::nn::ops {

class Op;
using OpPtr = std::shared_ptr<Op>;

}  // namespace thirdai::bolt::nn::ops

namespace thirdai::bolt::nn::tensor {

class Tensor {
 public:
  explicit Tensor(uint32_t dim);

  virtual BoltVector& getVector(uint32_t index) = 0;

  uint32_t dim() const;

  virtual std::optional<uint32_t> numNonzeros() const = 0;

  void addDependantOp(ops::OpPtr op);

  const std::vector<ops::OpPtr>& dependantOps() const;

 private:
  // TODO(Nicholas): Update this to support N dimensions (not required for V0).
  uint32_t _dim;

 protected:
  std::vector<ops::OpPtr> _dependant_ops;
};

using TensorPtr = std::shared_ptr<Tensor>;

}  // namespace thirdai::bolt::nn::tensor