#include "Input.h"
#include <cereal/archives/binary.hpp>
#include <cereal/archives/portable_binary.hpp>
#include <cereal/types/polymorphic.hpp>
#include <cereal/types/string.hpp>

namespace thirdai::bolt {

Input::Input(uint32_t expected_input_dim,
             std::optional<std::pair<uint32_t, uint32_t>> num_nonzeros_range)
    : _compiled(false),
      _input_batch(nullptr),
      _expected_input_dim(expected_input_dim),
      _num_nonzeros_range(std::move(num_nonzeros_range)) {}

void Input::setInputs(BoltBatch* inputs) {
  assert(inputs != nullptr);
  inputs->verifyExpectedDimension(
      /* expected_dimension = */ _expected_input_dim,
      /* num_nonzeros_range = */ _num_nonzeros_range,
      /* origin_string = */
      "We found an Input BoltVector larger than the expected input dim");
  _input_batch = inputs;
}

void Input::compileImpl() {
  if (_expected_input_dim == 0) {
    throw exceptions::GraphCompilationFailure(
        "Cannot have input layer with dimension 0.");
  }
  _compiled = true;
}

void Input::prepareForBatchProcessingImpl(uint32_t batch_size,
                                          bool use_sparsity) {
  (void)batch_size;
  (void)use_sparsity;
  throw exceptions::NodeStateMachineError(
      "Should never call prepareForBatchProcessing on Input (instead should "
      "call setInputs).");
}

void Input::summarizeImpl(std::stringstream& summary, bool detailed) const {
  (void)detailed;
  summary << name() << " (Input): dim=" << _expected_input_dim;
  if (_num_nonzeros_range) {
    summary << ", num_nonzeros_range=[" << _num_nonzeros_range->first << ","
            << _num_nonzeros_range->second << "]";
  }
  summary << "\n";
}

Node::NodeState Input::getState() const {
  if (!_compiled && _input_batch == nullptr) {
    return NodeState::PredecessorsSet;
  }
  if (_compiled && _input_batch == nullptr) {
    return NodeState::Compiled;
  }
  if (_compiled && _input_batch != nullptr) {
    return NodeState::PreparedForBatchProcessing;
  }
  throw exceptions::NodeStateMachineError(
      "InputNode is in an invalid internal state");
}

void Input::checkDimForInput(const BoltVector& vec) const {
  if (vec.isDense()) {
    if (vec.len != _expected_input_dim) {
      throw std::invalid_argument("Received dense BoltVector with dimension=" +
                                  std::to_string(vec.len) +
                                  " in input layer with dimension=" +
                                  std::to_string(_expected_input_dim));
    }
  } else {
    for (uint32_t i = 0; i < vec.len; i++) {
      uint32_t active_neuron = vec.active_neurons[i];
      if (active_neuron >= _expected_input_dim) {
        throw std::invalid_argument(
            "Received sparse BoltVector with active_neuron=" +
            std::to_string(active_neuron) + " in input layer with dimension=" +
            std::to_string(_expected_input_dim));
      }
    }
  }
}

template <class Archive>
void Input::serialize(Archive& archive) {
  archive(cereal::base_class<Node>(this), _compiled, _expected_input_dim);
}

template void Input::serialize<cereal::BinaryInputArchive>(
    cereal::BinaryInputArchive&);
template void Input::serialize<cereal::BinaryOutputArchive>(
    cereal::BinaryOutputArchive&);

template void Input::serialize<cereal::PortableBinaryInputArchive>(
    cereal::PortableBinaryInputArchive&);
template void Input::serialize<cereal::PortableBinaryOutputArchive>(
    cereal::PortableBinaryOutputArchive&);

}  // namespace thirdai::bolt

CEREAL_REGISTER_TYPE(thirdai::bolt::Input)
