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
  Tensor(uint32_t dim, bool sparse, std::optional<uint32_t> num_nonzeros)
      : _dim(dim), _sparse(sparse), _num_nonzeros(num_nonzeros) {
    if (!_sparse) {
      _num_nonzeros = _dim;
    }
  }

  virtual BoltVector& getVector(uint32_t index) = 0;

  uint32_t dim() const { return _dim; }

  std::optional<uint32_t> numNonzeros() const { return _num_nonzeros; }

  bool sparse() const { return _sparse; }

  void addDependantOp(ops::OpPtr op) {
    _dependant_ops.push_back(std::move(op));
  }

  const std::vector<ops::OpPtr>& dependantOps() const { return _dependant_ops; }

 private:
  // TODO(Nicholas): Update this to support N dimensions (not required for V0).
  uint32_t _dim;

 protected:
  bool _sparse;
  std::optional<uint32_t> _num_nonzeros;

  std::vector<ops::OpPtr> _dependant_ops;
};

using TensorPtr = std::shared_ptr<Tensor>;

}  // namespace thirdai::bolt::nn::tensor