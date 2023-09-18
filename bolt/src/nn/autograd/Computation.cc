#include "Computation.h"
#include <cereal/archives/binary.hpp>
#include <cereal/types/memory.hpp>
#include <cereal/types/vector.hpp>
#include <bolt/src/nn/ops/Op.h>
#include <stdexcept>
#include <string>

namespace thirdai::bolt {

Computation::Computation(OpPtr op, ComputationList inputs)
    : _op(std::move(op)), _inputs(std::move(inputs)) {}

ComputationPtr Computation::make(OpPtr op, ComputationList inputs) {
  return std::make_shared<Computation>(std::move(op), std::move(inputs));
}

OpPtr Computation::op() const { return _op; }

const ComputationList& Computation::inputs() const { return _inputs; }

TensorPtr& Computation::tensor() {
  assert(_output);
  return _output;
}

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
  auto nonzeros = _op->nonzeros(_inputs, use_sparsity);
  if (!nonzeros) {
    throw std::runtime_error(
        "Cannot allocate tensor for computation with unknown number of "
        "nonzeros.");
  }

  if (*nonzeros < dim && use_sparsity) {
    _output = Tensor::sparse(batch_size, dim, *nonzeros);
  } else {
    _output = Tensor::dense(batch_size, dim);
  }
}

void Computation::addInput(ComputationPtr input) {
  _inputs.push_back(std::move(input));
}

void Computation::setTensor(TensorPtr tensor) {
  if (tensor->dim() != dim()) {
    throw std::invalid_argument(
        "Cannot set tensor with dimension " + std::to_string(tensor->dim()) +
        " to computation with output dim " + std::to_string(dim()) +
        ". Op: " + _op->name() + " Computation: " + name());
  }
  _output = std::move(tensor);
}

void Computation::summary(std::ostream& summary) {
  _op->summary(summary, _inputs, this);
}

const std::string& Computation::name() const {
  if (_name.empty()) {
    throw std::runtime_error(
        "Attempted to access name of unnamed computation.");
  }
  return _name;
}

void Computation::setName(const std::string& name) {
  if (!_name.empty()) {
    throw std::runtime_error(
        "Computations should only be named by the model, and computations "
        "should not be reused between models.");
  }
  _name = name;
}

template void Computation::serialize(cereal::BinaryInputArchive&);
template void Computation::serialize(cereal::BinaryOutputArchive&);

template <class Archive>
void Computation::serialize(Archive& archive) {
  // Because inputs are also computations clang-tidy things this is an infinite
  // recursive loop because eventually the serialize function for the input
  // computations are called within the serialize function for this computation.
  archive(_op, _inputs, _name);  // NOLINT
}

}  // namespace thirdai::bolt