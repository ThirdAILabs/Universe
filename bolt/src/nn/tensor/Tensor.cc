#include "Tensor.h"

namespace thirdai::bolt::nn::tensor {

Tensor::Tensor(uint32_t dim, bool sparse, std::optional<uint32_t> num_nonzeros)
    : _dim(dim), _sparse(sparse), _num_nonzeros(num_nonzeros) {
  if (!_sparse) {
    _num_nonzeros = _dim;
  }
}

uint32_t Tensor::dim() const { return _dim; }

std::optional<uint32_t> Tensor::numNonzeros() const { return _num_nonzeros; }

bool Tensor::sparse() const { return _sparse; }

void Tensor::addDependantOp(ops::OpPtr op) {
  _dependant_ops.push_back(std::move(op));
}

const std::vector<ops::OpPtr>& Tensor::dependantOps() const {
  return _dependant_ops;
}

}  // namespace thirdai::bolt::nn::tensor