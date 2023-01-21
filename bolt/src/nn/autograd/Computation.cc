#include "Computation.h"
#include <bolt/src/nn/ops/Op.h>
#include <string>

namespace thirdai::bolt::nn::autograd {

std::string nextComputationName() {
  static uint32_t constructed = 0;
  return "tensor_" + std::to_string(++constructed);
}

Computation::Computation(ops::OpPtr op, ComputationList inputs)
    : _op(std::move(op)),
      _inputs(std::move(inputs)),
      _name(nextComputationName()) {}

ComputationPtr Computation::make(ops::OpPtr op, ComputationList inputs) {
  return std::make_shared<Computation>(std::move(op), std::move(inputs));
}

ops::OpPtr Computation::op() const { return _op; }

const ComputationList& Computation::inputs() const { return _inputs; }

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
  uint32_t nonzeros = _op->nonzeros(_inputs, use_sparsity).value();

  if (nonzeros < dim && use_sparsity) {
    _output = tensor::Tensor::sparse(batch_size, dim, nonzeros);
  } else {
    _output = tensor::Tensor::dense(batch_size, dim);
  }
}

void Computation::addInput(ComputationPtr input) {
  _inputs.push_back(std::move(input));
}

void Computation::summary(std::ostream& summary) {
  _op->summary(summary, _inputs, this);
}

const std::string& Computation::name() const { return _name; }

}  // namespace thirdai::bolt::nn::autograd