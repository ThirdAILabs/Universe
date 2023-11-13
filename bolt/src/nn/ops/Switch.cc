#include "Switch.h"
#include <cereal/archives/binary.hpp>
#include <cereal/types/base_class.hpp>
#include <cereal/types/memory.hpp>
#include <cereal/types/polymorphic.hpp>
#include <cereal/types/vector.hpp>
#include <bolt/src/nn/ops/FullyConnected.h>
#include <bolt/src/nn/ops/Op.h>
#include <archive/src/ArchiveList.h>
#include <archive/src/ArchiveMap.h>
#include <memory>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace thirdai::bolt {

std::string nextSwitchOpName() {
  static uint32_t constructed = 0;
  return "switch_" + std::to_string(++constructed);
}

Switch::Switch(uint32_t n_layers, uint32_t dim, uint32_t input_dim,
               float sparsity, const std::string& activation,
               const SamplingConfigPtr& sampling, bool use_bias,
               uint32_t rebuild_hash_tables,
               uint32_t reconstruct_hash_functions)
    : Op(nextSwitchOpName()) {
  for (uint32_t layer_id = 0; layer_id < n_layers; layer_id++) {
    _fc_ops.emplace_back(FullyConnected::make(
        dim, input_dim, sparsity, activation, sampling, use_bias,
        rebuild_hash_tables, reconstruct_hash_functions));
  }
}

std::shared_ptr<Switch> Switch::make(uint32_t n_layers, uint32_t dim,
                                     uint32_t input_dim, float sparsity,
                                     const std::string& activation,
                                     const SamplingConfigPtr& sampling,
                                     bool use_bias,
                                     uint32_t rebuild_hash_tables,
                                     uint32_t reconstruct_hash_functions) {
  return std::shared_ptr<Switch>(
      new Switch(n_layers, dim, input_dim, sparsity, activation, sampling,
                 use_bias, rebuild_hash_tables, reconstruct_hash_functions));
}

void Switch::forward(const ComputationList& inputs, TensorPtr& output,
                     uint32_t index_in_batch, bool training) {
  assert(inputs.size() == 2 || inputs.size() == 3);

  auto fc_inputs = fcInputs(inputs);
  getFcOpForInputs(inputs, index_in_batch)
      ->forward(fc_inputs, output, index_in_batch, training);
}

void Switch::backpropagate(ComputationList& inputs, TensorPtr& output,
                           uint32_t index_in_batch) {
  assert(inputs.size() == 2 || inputs.size() == 3);

  auto fc_inputs = fcInputs(inputs);
  getFcOpForInputs(inputs, index_in_batch)
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

std::optional<uint32_t> Switch::nonzeros(const ComputationList& inputs,
                                         bool use_sparsity) const {
  return _fc_ops.front()->nonzeros(inputs, use_sparsity);
}

void Switch::initOptimizer() {
  for (auto& op : _fc_ops) {
    op->initOptimizer();
  }
}

void Switch::disableSparseParameterUpdates() {
  for (auto& op : _fc_ops) {
    op->disableSparseParameterUpdates();
  }
}

void Switch::enableSparseParameterUpdates() {
  for (auto& op : _fc_ops) {
    op->enableSparseParameterUpdates();
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

std::vector<std::vector<float>*> Switch::parameters() {
  std::vector<std::vector<float>*> parameters;
  for (auto& op : _fc_ops) {
    auto op_parameters = op->parameters();
    parameters.insert(parameters.end(), op_parameters.begin(),
                      op_parameters.end());
  }
  return parameters;
}

ComputationPtr Switch::applyToInputs(const ComputationList& inputs) {
  if (inputs.size() != 2) {
    throw std::invalid_argument("Expected Switch op to have two inputs.");
  }
  return apply(inputs.at(0), inputs.at(1));
}

ar::ConstArchivePtr Switch::toArchive(bool with_optimizer) const {
  (void)with_optimizer;

  auto map = ar::ArchiveMap::make();

  map->set("name", ar::str(name()));
  map->set("type", ar::str("switch"));

  auto list = ar::ArchiveList::make();
  for (const auto& op : _fc_ops) {
    list->append(op->toArchive(with_optimizer));
  }
  map->set("fc_ops", list);

  return map;
}

void Switch::summary(std::ostream& summary, const ComputationList& inputs,
                     const Computation* output) const {
  summary << "Switch(" << name() << "): switch on " << inputs.front()->name()
          << std::endl;
  summary << "Contains: ";
  _fc_ops.front()->summary(summary, fcInputs(inputs), output);
  summary << std::endl;
}

void Switch::setSerializeOptimizer(bool should_serialize_optimizer) {
  for (const auto& op : _fc_ops) {
    op->setSerializeOptimizer(should_serialize_optimizer);
  }
}

ComputationPtr Switch::apply(ComputationPtr index, ComputationPtr input) {
  if (index->dim() != _fc_ops.size()) {
    std::stringstream error;
    error << "Cannot apply Switch op with n_layers = " << _fc_ops.size()
          << " to input tensor with dim " << index->dim() << ".";
    throw std::invalid_argument(error.str());
  }
  if (input->dim() != inputDim()) {
    std::stringstream error;
    error << "Cannot apply Switch op with weight matrix of shape (" << dim()
          << ", " << inputDim() << ") to input tensor with dim " << input->dim()
          << ".";

    throw std::invalid_argument(error.str());
  }
  return Computation::make(shared_from_this(),
                           {std::move(index), std::move(input)});
}

uint32_t Switch::inputDim() const { return _fc_ops.front()->inputDim(); }

void Switch::freezeHashTables(bool insert_labels_if_not_found) {
  for (auto& op : _fc_ops) {
    op->freezeHashTables(insert_labels_if_not_found);
  }
}

void Switch::setWeights(uint32_t layer_id, const float* weights_to_set) {
  getFcOpById(layer_id)->setWeights(weights_to_set);
}

void Switch::setBiases(uint32_t layer_id, const float* biases_to_set) {
  getFcOpById(layer_id)->setBiases(biases_to_set);
}

void Switch::autotuneRehashRebuild(uint32_t num_batches, uint32_t batch_size) {
  for (auto& op : _fc_ops) {
    op->autotuneRehashRebuild(num_batches, batch_size);
  }
}

FullyConnectedPtr Switch::getFcOpForInputs(const ComputationList& inputs,
                                           uint32_t index_in_batch) {
  uint32_t fc_id =
      inputs[0]->tensor()->getVector(index_in_batch).active_neurons[0];
  if (fc_id >= _fc_ops.size()) {
    throw std::runtime_error("Switch: FCID out of range " +
                             std::to_string(fc_id) +
                             "; size = " + std::to_string(_fc_ops.size()));
  }
  return _fc_ops[fc_id];
}

FullyConnectedPtr Switch::getFcOpById(uint32_t layer_id) {
  if (layer_id >= _fc_ops.size()) {
    std::stringstream error;
    error << "Tried to set weights and biases of layer_id=" << layer_id
          << " in Switch op with n_layers=" << _fc_ops.size() << ".";
    throw std::invalid_argument(error.str());
  }
  return _fc_ops[layer_id];
}

ComputationList Switch::fcInputs(const ComputationList& inputs) {
  return {inputs.begin() + 1, inputs.end()};
}

template void Switch::serialize(cereal::BinaryOutputArchive&);
template void Switch::serialize(cereal::BinaryInputArchive&);

template <class Archive>
void Switch::serialize(Archive& archive) {
  archive(cereal::base_class<Op>(this), _fc_ops);
}

}  // namespace thirdai::bolt

CEREAL_REGISTER_TYPE(thirdai::bolt::Switch)