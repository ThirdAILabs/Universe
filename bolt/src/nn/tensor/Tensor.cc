#include "Tensor.h"

namespace thirdai::bolt::nn::tensor {

Tensor::Tensor(uint32_t dim, std::string name)
    : _dim(dim), _name(std::move(name)) {}

uint32_t Tensor::dim() const { return _dim; }

const std::string& Tensor::name() const { return _name; }

}  // namespace thirdai::bolt::nn::tensor