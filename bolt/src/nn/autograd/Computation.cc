#include "Computation.h"
#include <cereal/archives/binary.hpp>
#include <cereal/types/memory.hpp>
#include <cereal/types/vector.hpp>
#include <bolt/src/nn/ops/Op.h>
#include <stdexcept>
#include <string>

namespace thirdai::bolt::nn::autograd {

std::string nextComputationName() {
  static uint32_t constructed = 0;
  /**
   * We name this tensor because there is a symmetry between tensors and
   * computations, each computation generates 1 tensor in the model. Calling it
   * tensor makes it more intuitive when model summaries are displayed and keeps
   * the computation class hidden from the user.
   */
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

tensor::TensorPtr& Computation::tensor() {
  assert(_output);
  return _output;
}

void Computation::forward(uint32_t index_in_batch, bool training) {
  _op->forward(_inputs, _output, index_in_batch, training);
}

void Computation::backpropagate(uint32_t index_in_batch) {
  _op->backpropagate(_inputs, _output, index_in_batch);
}

tensor::Dims Computation::dims() const { return _op->dims(_inputs); }

std::optional<uint32_t> Computation::nonzeros(bool use_sparsity) const {
  return _op->nonzeros(_inputs, use_sparsity);
}

void Computation::allocate(uint32_t batch_size, bool use_sparsity) {
  auto output_dims = this->dims();
  auto nonzeros = this->nonzeros(use_sparsity);
  if (!nonzeros) {
    throw std::runtime_error(
        "Cannot allocate tensor for computation with unknown number of "
        "nonzeros.");
  }

  tensor::Dims dims = {batch_size};
  dims.insert(dims.end(), output_dims.begin(), output_dims.end());

  if (*nonzeros < output_dims.back() && use_sparsity) {
    _output = tensor::Tensor::sparse(std::move(dims), *nonzeros);
  } else {
    _output = tensor::Tensor::dense(std::move(dims));
  }
}

void Computation::addInput(ComputationPtr input) {
  _inputs.push_back(std::move(input));
}

void Computation::setTensor(tensor::TensorPtr tensor) {
  tensor::Dims input_dims = tensor->dims();
  input_dims.erase(input_dims.begin());  // Erase batch size.

  if (tensor::areDimsEq(input_dims, dims())) {
    throw std::invalid_argument(
        "Cannot set tensor with dimensions " + tensor::toString(input_dims) +
        " to computation with dimensions " + tensor::toString(dims()) + ".");
  }
  _output = std::move(tensor);
}

void Computation::summary(std::ostream& summary) {
  _op->summary(summary, _inputs, this);
}

const std::string& Computation::name() const { return _name; }

template void Computation::serialize(cereal::BinaryInputArchive&);
template void Computation::serialize(cereal::BinaryOutputArchive&);

template <class Archive>
void Computation::serialize(Archive& archive) {
  // Because inputs are also computations clang-tidy things this is an infinite
  // recursive loop because eventually the serialize function for the input
  // computations are called within the serialize function for this computation.
  archive(_op, _inputs, _name);  // NOLINT
}

}  // namespace thirdai::bolt::nn::autograd