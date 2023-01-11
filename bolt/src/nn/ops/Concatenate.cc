#include "Concatenate.h"
#include <bolt/src/nn/tensor/ActivationTensor.h>
#include <bolt_vector/src/BoltVector.h>
#include <memory>
#include <numeric>
#include <stdexcept>

namespace thirdai::bolt::nn::ops {

std::string nextConcatenateOpName() {
  static uint32_t constructed = 0;
  return "concat_" + std::to_string(++constructed);
}

Concatenate::Concatenate() : Op(nextConcatenateOpName()) {}

std::shared_ptr<Concatenate> Concatenate::make() {
  return std::shared_ptr<Concatenate>(new Concatenate());
}

tensor::ActivationTensorPtr Concatenate::apply(
    const tensor::TensorList& inputs) {
  uint32_t total_output_dim = 0;
  if (_input_dims.empty()) {
    uint32_t current_neuron_offset = 0;
    for (const auto& input : inputs) {
      _input_dims.push_back(input->dim());
      _neuron_offsets.push_back(current_neuron_offset);
      current_neuron_offset += input->dim();
    }
    total_output_dim = current_neuron_offset;
  } else {
    if (inputs.size() != _input_dims.size()) {
      throw std::invalid_argument("Need better error message.");
    }
    for (uint32_t input_idx = 0; input_idx < inputs.size(); input_idx++) {
      if (inputs[input_idx]->dim() != _input_dims.at(input_idx)) {
        throw std::invalid_argument("Need better error message.");
      }
      total_output_dim += _input_dims[input_idx];
    }
  }

  return tensor::ActivationTensor::make(total_output_dim, shared_from_this(),
                                        inputs);
}

void Concatenate::forward(const tensor::TensorList& inputs,
                          tensor::ActivationTensor* output,
                          uint32_t index_in_batch, bool training) {
  (void)training;

  BoltVector& output_vector = output->getVector(index_in_batch);

  uint32_t current_offset_in_output = 0;

  for (uint32_t input_idx = 0; input_idx < inputs.size(); input_idx++) {
    const BoltVector& input_vector =
        inputs[input_idx]->getVector(index_in_batch);

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

void Concatenate::backpropagate(tensor::TensorList& inputs,
                                tensor::ActivationTensor* output,
                                uint32_t index_in_batch) {
  BoltVector& output_vector = output->getVector(index_in_batch);

  uint32_t current_offset_in_output = 0;

  for (auto& input : inputs) {
    BoltVector& input_vector = input->getVector(index_in_batch);

    if (input_vector.hasGradients()) {
      for (uint32_t i = 0; i < input_vector.len; i++) {
        input_vector.gradients[i] +=
            output_vector.gradients[i + current_offset_in_output];
      }
    }

    current_offset_in_output += input_vector.len;
  }
}

uint32_t Concatenate::numNonzerosInOutput(const tensor::TensorList& inputs,
                                          bool use_sparsity) const {
  uint32_t total_num_nonzeros = 0;
  for (const auto& input : inputs) {
    if (auto num_nonzeros = input->numNonzeros(use_sparsity)) {
      total_num_nonzeros += *num_nonzeros;
    } else {
      throw std::runtime_error(
          "Cannot concatenate input without specifying num_nonzeros.");
    }
  }

  return total_num_nonzeros;
}

void Concatenate::summary(std::ostream& summary,
                          const tensor::TensorList& inputs,
                          const tensor::ActivationTensor* output) const {
  summary << "Concatenate(" << name() << "): (";
  for (uint32_t i = 0; i < inputs.size(); i++) {
    summary << inputs[i]->name();
    if (i < inputs.size() - 1) {
      summary << ", ";
    }
  }
  summary << ") -> " << output->name();
}

}  // namespace thirdai::bolt::nn::ops