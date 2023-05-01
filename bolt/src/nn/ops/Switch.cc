#include "Switch.h"
#include <cereal/archives/binary.hpp>
#include <cereal/specialize.hpp>
#include <cereal/types/base_class.hpp>
#include <cereal/types/memory.hpp>
#include <cereal/types/polymorphic.hpp>
#include <cereal/types/vector.hpp>
#include <bolt/src/nn/ops/FullyConnected.h>
#include <memory>
#include <optional>
#include <vector>

namespace thirdai::bolt::nn::ops {

std::string nextSwitchOpName() {
  static uint32_t constructed = 0;
  return "switch_" + std::to_string(++constructed);
}

std::shared_ptr<Switch> Switch::make(uint32_t n_layers, uint32_t dim,
                                     uint32_t input_dim, float sparsity,
                                     const std::string& activation,
                                     const SamplingConfigPtr& sampling,
                                     uint32_t rebuild_hash_tables,
                                     uint32_t reconstruct_hash_functions) {
  return std::shared_ptr<Switch>(
      new Switch(n_layers, dim, input_dim, sparsity, activation, sampling,
                 rebuild_hash_tables, reconstruct_hash_functions));
}

void Switch::forward(const autograd::ComputationList& inputs,
                     tensor::TensorPtr& output, uint32_t index_in_batch,
                     bool training) {
  assert(inputs.size() == 2 || inputs.size() == 3);

  auto fc_inputs = fcInputs(inputs);
  fcOp(inputs, index_in_batch)
      ->forward(fc_inputs, output, index_in_batch, training);
}

void Switch::backpropagate(autograd::ComputationList& inputs,
                           tensor::TensorPtr& output, uint32_t index_in_batch) {
  assert(inputs.size() == 2 || inputs.size() == 3);

  auto fc_inputs = fcInputs(inputs);
  fcOp(inputs, index_in_batch)
      ->backpropagate(fc_inputs, output, index_in_batch);
}

void Switch::updateParameters(float learning_rate, uint32_t train_steps) {
  // TODO(Geordie): Technically each op will have around train_steps / n_layers
  // train steps. Is this going to be a problem?
  for (auto& op : _fc_ops) {
    op->updateParameters(learning_rate, train_steps);
  }
}

uint32_t Switch::dim() const { return _fc_ops.front()->dim(); }

std::optional<uint32_t> Switch::nonzeros(
    const autograd::ComputationList& inputs, bool use_sparsity) const {
  return _fc_ops.front()->nonzeros(inputs, use_sparsity);
}

void Switch::disableSparseParameterUpdates() {
  for (auto& op : _fc_ops) {
    op->disableSparseParameterUpdates();
  }
}

std::vector<std::vector<float>*> Switch::gradients() {
  std::vector<std::vector<float>*> gradients;
  for (auto& op : _fc_ops) {
    auto op_gradients = op->gradients();
    gradients.insert(gradients.end(), op_gradients.begin(), op_gradients.end());
  }
  return gradients;
}

void Switch::summary(std::ostream& summary,
                     const autograd::ComputationList& inputs,
                     const autograd::Computation* output) const {
  summary << "Switch(" << name() << "):" << std::endl;
  for (const auto& op : _fc_ops) {
    summary << "\t";
    op->summary(summary, inputs, output);
  }
}

void Switch::setSerializeOptimizer(bool should_serialize_optimizer) {
  for (const auto& op : _fc_ops) {
    op->setSerializeOptimizer(should_serialize_optimizer);
  }
}

autograd::ComputationPtr Switch::apply(autograd::ComputationPtr index,
                                       autograd::ComputationPtr input) {
  if (input->dim() != inputDim()) {
    std::stringstream error;
    error << "Cannot apply Switch op with weight matrix of shape (" << dim()
          << ", " << inputDim() << ") to input tensor with dim " << input->dim()
          << ".";

    throw std::invalid_argument(error.str());
  }
  return autograd::Computation::make(shared_from_this(),
                                     {std::move(index), std::move(input)});
}

uint32_t Switch::inputDim() const { return _fc_ops.front()->inputDim(); }

void Switch::freezeHashTables(bool insert_labels_if_not_found) {
  for (auto& op : _fc_ops) {
    op->freezeHashTables(insert_labels_if_not_found);
  }
}

void Switch::setWeightsAndBiases(const float* weights_to_set,
                                 const float* biases_to_set) {
  for (auto& op : _fc_ops) {
    op->setWeightsAndBiases(weights_to_set, biases_to_set);
  }
}

void Switch::autotuneRehashRebuild(uint32_t num_batches, uint32_t batch_size) {
  for (auto& op : _fc_ops) {
    op->autotuneRehashRebuild(num_batches, batch_size);
  }
}

Switch::Switch(uint32_t n_layers, uint32_t dim, uint32_t input_dim,
               float sparsity, const std::string& activation,
               const SamplingConfigPtr& sampling, uint32_t rebuild_hash_tables,
               uint32_t reconstruct_hash_functions)
    : Op(nextSwitchOpName()) {
  for (uint32_t layer_id = 0; layer_id < n_layers; layer_id++) {
    _fc_ops.emplace_back(
        FullyConnected::make(dim, input_dim, sparsity, activation, sampling,
                             rebuild_hash_tables, reconstruct_hash_functions));
  }
}

FullyConnectedPtr Switch::fcOp(const autograd::ComputationList& inputs,
                               uint32_t index_in_batch) {
  uint32_t fc_id =
      inputs[0]->tensor()->getVector(index_in_batch).active_neurons[0];
  // Default to last op if fc_id >= number of ops.
  fc_id = std::min<uint32_t>(fc_id, _fc_ops.size() - 1);
  return _fc_ops[fc_id];
}

autograd::ComputationList Switch::fcInputs(
    const autograd::ComputationList& inputs) {
  return {inputs.begin() + 1, inputs.end()};
}

template void Switch::serialize(cereal::BinaryOutputArchive&);
template void Switch::serialize(cereal::BinaryInputArchive&);

template <class Archive>
void Switch::serialize(Archive& archive) {
  archive(cereal::base_class<Op>(this), _fc_ops);
}

}  // namespace thirdai::bolt::nn::ops

CEREAL_REGISTER_TYPE(thirdai::bolt::nn::ops::Switch)