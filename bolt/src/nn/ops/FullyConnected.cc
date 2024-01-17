#include "FullyConnected.h"
#include <cereal/archives/binary.hpp>
#include <cereal/specialize.hpp>
#include <cereal/types/base_class.hpp>
#include <cereal/types/memory.hpp>
#include <cereal/types/polymorphic.hpp>
#include <bolt/src/layers/FullyConnectedLayer.h>
#include <bolt/src/layers/LayerUtils.h>
#include <bolt/src/nn/model/Model.h>
#include <bolt/src/nn/ops/Op.h>
#include <bolt/src/nn/tensor/Tensor.h>
#include <bolt_vector/src/BoltVector.h>
#include <archive/src/Archive.h>
#include <archive/src/Map.h>
#include <archive/src/ParameterReference.h>
#include <cstring>
#include <memory>
#include <stdexcept>

namespace thirdai::bolt {

std::string nextFullyConnectedOpName() {
  static uint32_t constructed = 0;
  return "fc_" + std::to_string(++constructed);
}

FullyConnected::FullyConnected(uint32_t dim, uint32_t input_dim, float sparsity,
                               const std::string& activation,
                               SamplingConfigPtr sampling, bool use_bias,
                               uint32_t rebuild_hash_tables,
                               uint32_t reconstruct_hash_functions)
    : Op(nextFullyConnectedOpName()),
      _rebuild_hash_tables(rebuild_hash_tables),
      _reconstruct_hash_functions(reconstruct_hash_functions),
      _updates_since_rebuild_hash_tables(0),
      _updates_since_reconstruct_hash_functions(0) {
  if (!sampling) {
    sampling = DWTASamplingConfig::autotune(dim, sparsity,
                                            /* experimental_autotune=*/false);
  }
  FullyConnectedLayerConfig config(dim, sparsity, activation,
                                   std::move(sampling));

  _kernel = std::make_shared<FullyConnectedLayer>(
      config, input_dim, /* disable_sparse_sparse_updates */ false, use_bias);
}

std::shared_ptr<FullyConnected> FullyConnected::make(
    uint32_t dim, uint32_t input_dim, float sparsity,
    const std::string& activation, SamplingConfigPtr sampling, bool use_bias,
    uint32_t rebuild_hash_tables, uint32_t reconstruct_hash_functions) {
  return std::shared_ptr<FullyConnected>(new FullyConnected(
      dim, input_dim, sparsity, activation, std::move(sampling), use_bias,
      rebuild_hash_tables, reconstruct_hash_functions));
}

void FullyConnected::forward(const ComputationList& inputs, TensorPtr& output,
                             uint32_t index_in_batch, bool training) {
  assert(inputs.size() == 1 || inputs.size() == 2);
  // If the op is an output pass in labels during training to ensure labels are
  // in active neuron set.
  const BoltVector* labels = nullptr;
  if (training && inputs.size() == 2) {
    labels = &inputs[1]->tensor()->getVector(index_in_batch);
  }
  _kernel->forward(inputs[0]->tensor()->getVector(index_in_batch),
                   output->getVector(index_in_batch), labels);
}

void FullyConnected::backpropagate(ComputationList& inputs, TensorPtr& output,
                                   uint32_t index_in_batch) {
  assert(inputs.size() == 1 || inputs.size() == 2);

  BoltVector& input = inputs[0]->tensor()->getVector(index_in_batch);

  if (input.hasGradients()) {
    _kernel->backpropagate(input, output->getVector(index_in_batch));
  } else {
    _kernel->backpropagateInputLayer(input, output->getVector(index_in_batch));
  }
}

void FullyConnected::updateParameters(float learning_rate,
                                      uint32_t train_steps) {
  _kernel->updateParameters(learning_rate, train_steps, BETA1, BETA2, EPS);

  if (++_updates_since_reconstruct_hash_functions ==
      _reconstruct_hash_functions) {
    _kernel->reBuildHashFunction();

    _updates_since_rebuild_hash_tables = 0;
    _updates_since_reconstruct_hash_functions = 0;
  } else if (++_updates_since_rebuild_hash_tables == _rebuild_hash_tables) {
    _kernel->buildHashTables();
    _updates_since_rebuild_hash_tables = 0;
  }
}

uint32_t FullyConnected::dim() const { return _kernel->getDim(); }

std::optional<uint32_t> FullyConnected::nonzeros(const ComputationList& inputs,
                                                 bool use_sparsity) const {
  // The number of output nonzeros for a FullyConnected op do not depend on its
  // inputs.
  (void)inputs;
  if (use_sparsity) {
    return _kernel->getSparseDim();
  }
  return _kernel->getDim();
}

void FullyConnected::initOptimizer() { _kernel->initOptimizer(); }

void FullyConnected::disableSparseParameterUpdates() {
  _kernel->disableSparseParameterUpdates();
}

void FullyConnected::enableSparseParameterUpdates() {
  _kernel->enableSparseParameterUpdates();
}

std::vector<std::vector<float>*> FullyConnected::gradients() {
  return {&_kernel->weightsGradient(), &_kernel->biasGradient()};
}

std::vector<std::vector<float>*> FullyConnected::parameters() {
  return {&_kernel->weights(), &_kernel->biases()};
}

ComputationPtr FullyConnected::applyToInputs(const ComputationList& inputs) {
  // Can have a second input for labels if the output layer. This is only passed
  // into the apply method when rebuilding the computation graph after
  // deserialization. We can just ignore it here because the model will match
  // the sparse output layer with the corresponding labels.
  if (inputs.size() != 1 && inputs.size() != 2) {
    throw std::invalid_argument(
        "Expected FullyConnected op to have one or input.");
  }
  return apply(inputs.at(0));
}

ar::ConstArchivePtr FullyConnected::toArchive(bool with_optimizer) const {
  (void)with_optimizer;

  auto map = ar::Map::make();
  map->set("name", ar::str(name()));
  map->set("type", ar::str(type()));

  map->set("dim", ar::u64(dim()));
  map->set("input_dim", ar::u64(inputDim()));
  map->set("sparsity", ar::f32(_kernel->_sparsity));
  map->set("activation", ar::str(activationFunctionToStr(_kernel->_act_func)));
  map->set("use_bias", ar::boolean(_kernel->_use_bias));

  map->set("weights",
           ar::ParameterReference::make(_kernel->_weights, shared_from_this()));
  map->set("biases",
           ar::ParameterReference::make(_kernel->_biases, shared_from_this()));

  map->set("neuron_index", neuronIndexToArchive(_kernel->neuronIndex()));
  map->set("index_frozen", ar::boolean(_kernel->_index_frozen));
  map->set("rebuild_hash_tables", ar::u64(_rebuild_hash_tables));
  map->set("reconstruct_hash_functions", ar::u64(_reconstruct_hash_functions));

  if (with_optimizer && _kernel->_weight_optimizer &&
      _kernel->_bias_optimizer) {
    map->set("weight_opt",
             optimizerToArchive(*_kernel->_weight_optimizer, shared_from_this(),
                                dim(), inputDim()));

    map->set("bias_opt",
             optimizerToArchive(*_kernel->_bias_optimizer, shared_from_this(),
                                /*rows=*/1, dim()));
  }

  map->set("disable_sparse_parameter_updates",
           ar::boolean(_kernel->_disable_sparse_parameter_updates));

  return map;
}

std::shared_ptr<FullyConnected> FullyConnected::fromArchive(
    const ar::Archive& archive) {
  return std::shared_ptr<FullyConnected>(new FullyConnected(archive));
}

FullyConnected::FullyConnected(const ar::Archive& archive)
    : Op(archive.str("name")),
      _kernel(std::make_shared<FullyConnectedLayer>(archive)),
      _rebuild_hash_tables(archive.u64("rebuild_hash_tables")),
      _reconstruct_hash_functions(archive.u64("reconstruct_hash_functions")),
      _updates_since_rebuild_hash_tables(0),
      _updates_since_reconstruct_hash_functions(0) {}

void FullyConnected::summary(std::ostream& summary,
                             const ComputationList& inputs,
                             const Computation* output) const {
  summary << "FullyConnected(" << name() << "): " << inputs[0]->name() << " -> "
          << output->name();
  summary << " [dim=" << _kernel->getDim()
          << ", sparsity=" << _kernel->getSparsity() << ", activation="
          << activationFunctionToStr(_kernel->getActivationFunction());
  if (!_kernel->useBias()) {
    summary << ", bias=" << std::boolalpha << _kernel->useBias();
  }
  if (_kernel->getSparsity() < 1.0) {
    summary << ", sampling=(";
    _kernel->buildSamplingSummary(summary);
    summary << ", rebuild_hash_tables=" << _rebuild_hash_tables;
    summary << ", reconstruct_hash_functions=" << _reconstruct_hash_functions;
    summary << ")";
  }
  summary << "]";
}

void FullyConnected::setSerializeOptimizer(bool should_serialize_optimizer) {
  _kernel->saveWithOptimizer(should_serialize_optimizer);
}

void FullyConnected::reBuildHashFunction() { _kernel->reBuildHashFunction(); }
void FullyConnected::registerModel(const std::weak_ptr<Model>& new_model) {
  bool found = false;

  // This adds the new model to the list of models that the fully connected
  // layer is used in. This is so that if the sparsity of the layer is updated
  // and the model's internal state needs to be reallocated it can call the
  // appropriate method on the model to do so.
  for (const auto& model_wp : _models_using_op) {
    if (auto model = model_wp.lock()) {
      if (model == new_model.lock()) {
        found = true;
        break;
      }
    }
  }

  if (!found) {
    _models_using_op.push_back(new_model);
  }
}

std::vector<std::pair<std::string, double>>
FullyConnected::parameterAndGradNorms() const {
  std::vector<std::pair<std::string, double>> all_norms;

  computeNorms(_kernel->weights(), "weight", all_norms);
  if (_kernel->hasOptimizers()) {
    computeNorms(_kernel->weightsGradient(), "weight_grad", all_norms);
  }

  if (_kernel->useBias()) {
    computeNorms(_kernel->biases(), "bias", all_norms);
    if (_kernel->hasOptimizers()) {
      computeNorms(_kernel->biasGradient(), "bias_grad", all_norms);
    }
  }

  return all_norms;
}

ComputationPtr FullyConnected::apply(ComputationPtr input) {
  if (input->dim() != _kernel->getInputDim()) {
    std::stringstream error;
    error << "Cannot apply FullyConnected op with weight matrix of shape ("
          << _kernel->getDim() << ", " << _kernel->getInputDim()
          << ") to input tensor with dim " << input->dim() << ".";

    throw std::invalid_argument(error.str());
  }
  return Computation::make(shared_from_this(), {std::move(input)});
}

uint32_t FullyConnected::inputDim() const { return _kernel->getInputDim(); }

const float* FullyConnected::weightsPtr() const {
  return _kernel->getWeightsPtr();
}

const float* FullyConnected::biasesPtr() const {
  return _kernel->getBiasesPtr();
}

std::shared_ptr<FullyConnectedLayer> FullyConnected::kernel() const {
  return _kernel;
}

void FullyConnected::freezeHashTables(bool insert_labels_if_not_found) {
  _kernel->freezeHashTables(insert_labels_if_not_found);
}

void FullyConnected::unfreezeHashTables() { _kernel->unfreezeHashTables(); }

void FullyConnected::setWeights(const float* weights) {
  _kernel->setWeights(weights);
}

void FullyConnected::setBiases(const float* new_biases) {
  _kernel->setBiases(new_biases);
}

std::pair<hashing::HashFunctionPtr, hashtable::SampledHashTablePtr>
FullyConnected::getHashTable() const {
  return _kernel->getHashTable();
}

void FullyConnected::setHashTable(hashing::HashFunctionPtr hash_fn,
                                  hashtable::SampledHashTablePtr hash_table) {
  return _kernel->setHashTable(std::move(hash_fn), std::move(hash_table));
}

void FullyConnected::autotuneRehashRebuild(uint32_t num_batches,
                                           uint32_t batch_size) {
  // TODO(Someone): Revisit this autotuning. It seems like for some datasets it
  // will update too frequently, for instance 50 batches with a batch size of 2K
  // will lead to updates every batch.
  _reconstruct_hash_functions = std::max(num_batches / 4, 1U);

  if (num_batches * batch_size >= 100000) {
    _rebuild_hash_tables = std::max(num_batches / 100, 1U);
  } else {
    _rebuild_hash_tables = std::max(num_batches / 20, 1U);
  }
}

void FullyConnected::setSparsity(float sparsity, bool rebuild_hash_tables,
                                 bool experimental_autotune) {
  _kernel->setSparsity(sparsity, rebuild_hash_tables, experimental_autotune);

  // We need to the state to be reallocated after updating the sparsity. If a
  // sparsity is increased between processing batches of the same batch size,
  // both using sparsity. Then there will otherwise be no reallocation of state
  // for activations, and the existing allocated state will not be large enough
  // for the increased sparsity.
  for (auto& model_wp : _models_using_op) {
    if (auto model = model_wp.lock()) {
      model->forceStateReallocation();
    }
  }
}

template void FullyConnected::save(cereal::BinaryOutputArchive&) const;

template <class Archive>
void FullyConnected::save(Archive& archive) const {
  archive(cereal::base_class<Op>(this), _kernel, _rebuild_hash_tables,
          _reconstruct_hash_functions, _updates_since_rebuild_hash_tables,
          _updates_since_reconstruct_hash_functions);
}

template void FullyConnected::load(cereal::BinaryInputArchive&);

template <class Archive>
void FullyConnected::load(Archive& archive) {
  archive(cereal::base_class<Op>(this), _kernel, _rebuild_hash_tables,
          _reconstruct_hash_functions, _updates_since_rebuild_hash_tables,
          _updates_since_reconstruct_hash_functions);
}

}  // namespace thirdai::bolt

CEREAL_REGISTER_TYPE_WITH_NAME(thirdai::bolt::FullyConnected,
                               "thirdai::bolt::nn::ops::FullyConnected")
