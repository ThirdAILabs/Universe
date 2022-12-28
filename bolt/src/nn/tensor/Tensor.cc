#include "Tensor.h"

namespace thirdai::bolt::nn::tensor {

Tensor::Tensor(uint32_t dim) : _dim(dim) {}

uint32_t Tensor::dim() const { return _dim; }

void Tensor::addDependantOp(ops::OpPtr op) {
  _dependant_ops.push_back(std::move(op));
}

const std::vector<ops::OpPtr>& Tensor::dependantOps() const {
  return _dependant_ops;
}

}  // namespace thirdai::bolt::nn::tensor