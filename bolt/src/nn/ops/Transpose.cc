#include "Transpose.h"
#include <cereal/archives/binary.hpp>
#include <cereal/types/base_class.hpp>
#include <cereal/types/polymorphic.hpp>
#include <bolt/src/nn/autograd/Computation.h>
#include <bolt/src/nn/tensor/Tensor.h>
#include <bolt_vector/src/BoltVector.h>
#include <bolt_vector/src/BoltVectorUtils.h>
#include <cmath>
#include <memory>
#include <stdexcept>

namespace thirdai::bolt::nn::ops {

std::string nextTransposeOpName() {
  static uint32_t constructed = 0;
  return "transpose_" + std::to_string(++constructed);
}

/**
 * Operates on an Input BoltVector which is in a row major format. This takes
 * the transpose of the BoltVector that is convert it into column major format.
 * rows : Number of rows in input vector
 * columns : Number of columns in input vector
 */
Transpose::Transpose(uint32_t rows, uint32_t columns)
    : Op(nextTransposeOpName()), _rows(rows), _columns(columns) {}

std::shared_ptr<Transpose> Transpose::make(uint32_t rows, uint32_t columns) {
  return std::shared_ptr<Transpose>(new Transpose(rows, columns));
}

void Transpose::forward(const autograd::ComputationList& inputs,
                        tensor::TensorPtr& output, uint32_t index_in_batch,
                        bool training) {
  (void)training;

  BoltVector& input_vector = inputs[0]->tensor()->getVector(index_in_batch);
  BoltVector& output_vector = output->getVector(index_in_batch);

  bolt_vector::transposeBoltVector(input_vector, output_vector, _rows,
                                   _columns);
}

void Transpose::backpropagate(autograd::ComputationList& inputs,
                              tensor::TensorPtr& output,
                              uint32_t index_in_batch) {
  BoltVector& input_vector = inputs[0]->tensor()->getVector(index_in_batch);
  BoltVector& output_vector = output->getVector(index_in_batch);

  bolt_vector::transposeBoltVector(output_vector, input_vector, _columns,
                                   _rows);
}

void Transpose::updateParameters(float learning_rate, uint32_t train_steps) {
  (void)learning_rate;
  (void)train_steps;
}

uint32_t Transpose::dim() const { return _rows * _columns; }

std::optional<uint32_t> Transpose::nonzeros(
    const autograd::ComputationList& inputs, bool use_sparsity) const {
  return inputs.at(0)->nonzeros(use_sparsity);
}

void Transpose::disableSparseParameterUpdates() {}

void Transpose::enableSparseParameterUpdates() {}

std::vector<std::vector<float>*> Transpose::gradients() { return {}; }
std::vector<std::vector<float>*> Transpose::parameters() { return {}; }

void Transpose::summary(std::ostream& summary,
                        const autograd::ComputationList& inputs,
                        const autograd::Computation* output) const {
  summary << "Transpose(" << name() << "): " << inputs[0]->name() << " -> "
          << output->name();

  summary << "[rows=" << _rows << ", columns=" << _columns << "]";
}

autograd::ComputationPtr Transpose::apply(autograd::ComputationPtr input) {
  if (_rows * _columns != input->dim()) {
    std::string error_report =
        "Cannot apply Input with dim: " + std::to_string(input->dim()) +
        "to Transpose layer with rows and columns: " + std::to_string(_rows) +
        " " + std::to_string(_columns);
    throw std::logic_error(error_report);
  }

  return autograd::Computation::make(shared_from_this(), {std::move(input)});
}

template void Transpose::serialize(cereal::BinaryInputArchive&);
template void Transpose::serialize(cereal::BinaryOutputArchive&);

template <class Archive>
void Transpose::serialize(Archive& archive) {
  archive(cereal::base_class<Op>(this), _rows, _columns);
}

}  // namespace thirdai::bolt::nn::ops

CEREAL_REGISTER_TYPE(thirdai::bolt::nn::ops::Transpose)