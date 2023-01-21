#include "Computation.h"
#include <bolt/src/nn/ops/Op.h>

namespace thirdai::bolt::nn::autograd {

Computation::Computation(ops::OpPtr op,
                         std::vector<std::shared_ptr<Computation>> inputs)
    : _op(std::move(op)), _inputs(std::move(inputs)) {}

ops::OpPtr Computation::op() const { return _op; }

const std::vector<std::shared_ptr<Computation>>& Computation::inputs() const {
  return _inputs;
}

tensor::TensorPtr& Computation::tensor() { return _output; }

void Computation::forward(uint32_t index_in_batch, bool training) {
  _op->forward(_inputs, _output, index_in_batch, training);
}

void Computation::backpropagate(uint32_t index_in_batch) {
  _op->backpropagate(_inputs, _output, index_in_batch);
}

uint32_t Computation::dim() const { return _op->dim(); }

std::optional<uint32_t> Computation::nonzeros(bool use_sparsity) const {
  return _op->nonzeros(_inputs, use_sparsity);
}

void Computation::allocate(uint32_t batch_size, bool use_sparsity) {
  uint32_t dim = _op->dim();
  uint32_t nonzeros = _op->nonzeros(_inputs, use_sparsity);

  if (nonzeros < dim) {
    // Allocate sparse tensor
    (void)batch_size;
  } else {
    // Allocate dense tensor
  }
}

void Computation::addInput(std::shared_ptr<Computation> input) {
  _inputs.push_back(std::move(input));
}

void Computation::summary(std::ostream& summary) {
  _op->summary(summary, _inputs, _output);
}

}  // namespace thirdai::bolt::nn::autograd