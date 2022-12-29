#include "Tensor.h"

namespace thirdai::bolt::nn::tensor {

Tensor::Tensor(uint32_t dim, std::string name)
    : _dim(dim), _name(std::move(name)) {}

uint32_t Tensor::dim() const { return _dim; }

void Tensor::addDependantOp(ops::OpPtr op) {
  _dependant_ops.push_back(std::move(op));
}

const std::vector<ops::OpPtr>& Tensor::dependantOps() const {
  return _dependant_ops;
}

const std::string& Tensor::name() const { return _name; }

}  // namespace thirdai::bolt::nn::tensor