#include "Concatenate.h"
#include <cereal/archives/binary.hpp>
#include <cereal/types/base_class.hpp>
#include <cereal/types/polymorphic.hpp>
#include <cereal/types/vector.hpp>
#include <bolt/src/nn/autograd/Computation.h>
#include <bolt/src/nn/ops/Op.h>
#include <bolt/src/nn/tensor/Tensor.h>
#include <memory>
#include <numeric>
#include <stdexcept>

namespace thirdai::bolt {

std::string nextConcatenateOpName() {
  static uint32_t constructed = 0;
  return "concat_" + std::to_string(++constructed);
}

Concatenate::Concatenate() : Op(nextConcatenateOpName()) {}

std::shared_ptr<Concatenate> Concatenate::make() {
  return std::shared_ptr<Concatenate>(new Concatenate());
}

void Concatenate::forward(const ComputationList& inputs, TensorPtr& output,
                          uint32_t index_in_batch, bool training) {
  (void)training;
  assert(!inputs.empty());

  BoltVector& output_vector = output->getVector(index_in_batch);

  uint32_t current_offset_in_output = 0;

  for (uint32_t input_idx = 0; input_idx < inputs.size(); input_idx++) {
    const BoltVector& input_vector =
        inputs[input_idx]->tensor()->getVector(index_in_batch);

    std::copy(input_vector.activations,
              input_vector.activations + input_vector.len,
              output_vector.activations + current_offset_in_output);

    if (!output_vector.isDense()) {
      if (input_vector.isDense()) {
        uint32_t* start =
            output_vector.active_neurons + current_offset_in_output;

        std::iota(start, start + input_vector.len, _neuron_offsets[input_idx]);
      } else {
        for (uint32_t i = 0; i < input_vector.len; i++) {
          output_vector.active_neurons[i + current_offset_in_output] =
              input_vector.active_neurons[i] + _neuron_offsets[input_idx];
        }
      }
    }

    current_offset_in_output += input_vector.len;
  }

  assert(current_offset_in_output == output_vector.len);
}

void Concatenate::backpropagate(ComputationList& inputs, TensorPtr& output,
                                uint32_t index_in_batch) {
  assert(!inputs.empty());

  BoltVector& output_vector = output->getVector(index_in_batch);

  uint32_t current_offset_in_output = 0;

  for (auto& input : inputs) {
    BoltVector& input_vector = input->tensor()->getVector(index_in_batch);

    if (input_vector.hasGradients()) {
      for (uint32_t i = 0; i < input_vector.len; i++) {
        input_vector.gradients[i] +=
            output_vector.gradients[i + current_offset_in_output];
      }
    }

    current_offset_in_output += input_vector.len;
  }
}

uint32_t Concatenate::dim() const {
  uint32_t total_dim = 0;
  for (uint32_t dim : _input_dims) {
    total_dim += dim;
  }
  return total_dim;
}

std::optional<uint32_t> Concatenate::nonzeros(const ComputationList& inputs,
                                              bool use_sparsity) const {
  uint32_t total_num_nonzeros = 0;
  for (const auto& input : inputs) {
    if (auto num_nonzeros = input->nonzeros(use_sparsity)) {
      total_num_nonzeros += *num_nonzeros;
    } else {
      throw std::runtime_error(
          "Cannot concatenate input without specifying num_nonzeros.");
    }
  }

  return total_num_nonzeros;
}

void Concatenate::initOptimizer() {}

void Concatenate::summary(std::ostream& summary, const ComputationList& inputs,
                          const Computation* output) const {
  summary << "Concatenate(" << name() << "): (";
  for (uint32_t i = 0; i < inputs.size(); i++) {
    summary << inputs[i]->name();
    if (i < inputs.size() - 1) {
      summary << ", ";
    }
  }
  summary << ") -> " << output->name();
}

ComputationPtr Concatenate::apply(const ComputationList& inputs) {
  if (_input_dims.empty()) {
    uint32_t current_neuron_offset = 0;
    for (const auto& input : inputs) {
      _input_dims.push_back(input->dim());
      _neuron_offsets.push_back(current_neuron_offset);
      current_neuron_offset += input->dim();
    }
  } else {
    if (inputs.size() != _input_dims.size()) {
      std::stringstream error;
      error << "Cannot apply Concatenate op expecting " << _input_dims.size()
            << " inputs to " << inputs.size() << " inputs.";
      throw std::invalid_argument(error.str());
    }
    for (uint32_t input_idx = 0; input_idx < inputs.size(); input_idx++) {
      if (inputs[input_idx]->dim() != _input_dims.at(input_idx)) {
        std::stringstream error;
        error << "Expected input at index " << input_idx
              << " to Concatenate op to have dim " << _input_dims.at(input_idx)
              << " but received input with dim " << inputs[input_idx]->dim()
              << ".";
        throw std::invalid_argument(error.str());
      }
    }
  }

  return Computation::make(shared_from_this(), inputs);
}

template void Concatenate::serialize(cereal::BinaryInputArchive&);
template void Concatenate::serialize(cereal::BinaryOutputArchive&);

template <class Archive>
void Concatenate::serialize(Archive& archive) {
  archive(cereal::base_class<Op>(this), _input_dims, _neuron_offsets);
}

}  // namespace thirdai::bolt

CEREAL_REGISTER_TYPE_WITH_NAME(thirdai::bolt::Concatenate,
                               "thirdai::bolt::nn::ops::Concatenate")