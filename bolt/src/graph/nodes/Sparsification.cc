#include "Sparsification.h"
#include <cereal/archives/binary.hpp>
#include <cereal/types/base_class.hpp>
#include <cereal/types/memory.hpp>
#include <cereal/types/optional.hpp>
#include <queue>

namespace thirdai::bolt {

struct NeuronInfo {
  NeuronInfo(uint32_t neuron, float activation)
      : neuron(neuron),
        activation(activation),
        activation_magnitude(abs(activation)) {}

  uint32_t neuron;
  float activation;
  float activation_magnitude;
};

struct NeuronInfoCmp {
  bool operator()(const NeuronInfo& a, const NeuronInfo& b) {
    return a.activation_magnitude > b.activation_magnitude;
  }
};

using NeuronQueue =
    std::priority_queue<NeuronInfo, std::vector<NeuronInfo>, NeuronInfoCmp>;

void SparsificationNode::forwardImpl(uint32_t vec_index,
                                     const BoltVector* labels) {
  (void)labels;

  const BoltVector& input = _input->getOutputVector(vec_index);
  BoltVector& output = (*_outputs)[vec_index];

  if (output.len < input.len) {
    NeuronQueue queue;
    for (uint32_t i = 0; i < output.len; i++) {
      queue.emplace(i, input.activations[i]);
    }

    for (uint32_t i = output.len; i < input.len; i++) {
      NeuronInfo info(i, input.activations[i]);
      if (info.activation_magnitude > queue.top().activation_magnitude) {
        queue.pop();
        queue.push(info);
      }
    }

    uint32_t index = 0;
    while (!queue.empty()) {
      output.active_neurons[index] = queue.top().neuron;
      output.activations[index] = queue.top().activation;
      queue.pop();
      index++;
    }
  } else {
    std::copy(input.activations, input.activations + input.len,
              output.activations);
  }
}

void SparsificationNode::backpropagateImpl(uint32_t vec_index) {
  const BoltVector& input = _input->getOutputVector(vec_index);
  BoltVector& output = (*_outputs)[vec_index];

  if (output.len < input.len) {
    for (uint32_t i = 0; i < output.len; i++) {
      input.gradients[output.active_neurons[i]] += output.gradients[i];
    }
  } else {
    std::copy(output.gradients, output.gradients + output.len, input.gradients);
  }
}

Node::NodeState SparsificationNode::getState() const {
  if (!_input && !_compiled && !_outputs) {
    return NodeState::Constructed;
  }
  if (_input && !_compiled && !_outputs) {
    return NodeState::PredecessorsSet;
  }
  if (_input && _compiled && !_outputs) {
    return NodeState::Compiled;
  }
  if (_input && _compiled && _outputs) {
    return NodeState::PreparedForBatchProcessing;
  }
  throw exceptions::NodeStateMachineError(
      "SparsificationNode is in an invalid internal state.");
}

template void SparsificationNode::serialize(cereal::BinaryInputArchive&);
template void SparsificationNode::serialize(cereal::BinaryOutputArchive&);

template <class Archive>
void SparsificationNode::serialize(Archive& archive) {
  archive(cereal::base_class<Node>(this), _compiled, _input, _sparsity,
          _outputs);
}

}  // namespace thirdai::bolt

CEREAL_REGISTER_TYPE(thirdai::bolt::SparsificationNode)