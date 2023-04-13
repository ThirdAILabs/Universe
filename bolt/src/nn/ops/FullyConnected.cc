#include "FullyConnected.h"
#include <cereal/archives/binary.hpp>
#include <cereal/specialize.hpp>
#include <cereal/types/base_class.hpp>
#include <cereal/types/memory.hpp>
#include <cereal/types/polymorphic.hpp>
#include <bolt/src/layers/LayerUtils.h>
#include <bolt/src/nn/ops/Op.h>
#include <bolt/src/nn/tensor/Tensor.h>
#include <bolt_vector/src/BoltVector.h>
#include <cstring>
#include <memory>
#include <stdexcept>

namespace thirdai::bolt::nn::ops {

std::string nextFullyConnectedOpName() {
  static uint32_t constructed = 0;
  return "fc_" + std::to_string(++constructed);
}

FullyConnected::FullyConnected(uint32_t dim, uint32_t input_dim, float sparsity,
                               const std::string& activation,
                               SamplingConfigPtr sampling,
                               uint32_t rebuild_hash_tables,
                               uint32_t reconstruct_hash_functions)
    : Op(nextFullyConnectedOpName()),
      _rebuild_hash_tables(rebuild_hash_tables),
      _reconstruct_hash_functions(reconstruct_hash_functions),
      _updates_since_rebuild_hash_tables(0),
      _updates_since_reconstruct_hash_functions(0) {
  if (!sampling) {
    sampling = DWTASamplingConfig::autotune(dim, sparsity);
  }
  FullyConnectedLayerConfig config(dim, sparsity, activation,
                                   std::move(sampling));

  _kernel = std::make_shared<FullyConnectedLayer>(config, input_dim);
}

std::shared_ptr<FullyConnected> FullyConnected::make(
    uint32_t dim, uint32_t input_dim, float sparsity,
    const std::string& activation, SamplingConfigPtr sampling,
    uint32_t rebuild_hash_tables, uint32_t reconstruct_hash_functions) {
  return std::shared_ptr<FullyConnected>(new FullyConnected(
      dim, input_dim, sparsity, activation, std::move(sampling),
      rebuild_hash_tables, reconstruct_hash_functions));
}

void FullyConnected::forward(const autograd::ComputationList& inputs,
                             tensor::TensorPtr& output, uint32_t index_in_batch,
                             bool training) {
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

void FullyConnected::backpropagate(autograd::ComputationList& inputs,
                                   tensor::TensorPtr& output,
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
    _kernel->buildHashTables();

    _updates_since_rebuild_hash_tables = 0;
    _updates_since_reconstruct_hash_functions = 0;
  } else if (++_updates_since_rebuild_hash_tables == _rebuild_hash_tables) {
    _kernel->buildHashTables();
    _updates_since_rebuild_hash_tables = 0;
  }
}

uint32_t FullyConnected::dim() const { return _kernel->getDim(); }

std::optional<uint32_t> FullyConnected::nonzeros(
    const autograd::ComputationList& inputs, bool use_sparsity) const {
  // The number of output nonzeros for a FullyConnected op do not depend on its
  // inputs.
  (void)inputs;
  if (use_sparsity) {
    return _kernel->getSparseDim();
  }
  return _kernel->getDim();
}

void FullyConnected::disableSparseParameterUpdates() {
  _kernel->disableSparseParameterUpdates();
}

std::vector<std::vector<float>*> FullyConnected::gradients() {
  return {&_kernel->weightsGradient(), &_kernel->biasGradient()};
}

void FullyConnected::summary(std::ostream& summary,
                             const autograd::ComputationList& inputs,
                             const autograd::Computation* output) const {
  summary << "FullyConnected(" << name() << "): " << inputs[0]->name() << " -> "
          << output->name();
  summary << " [dim=" << _kernel->getDim()
          << ", sparsity=" << _kernel->getSparsity() << ", activation="
          << activationFunctionToStr(_kernel->getActivationFunction());
  if (_kernel->getSparsity() < 1.0) {
    summary << ", sampling=(";
    _kernel->buildSamplingSummary(summary);
    summary << ", rebuild_hash_tables=" << _rebuild_hash_tables;
    summary << ", reconstruct_hash_functions=" << _reconstruct_hash_functions;
    summary << ")";
  }
  summary << "]";
}

autograd::ComputationPtr FullyConnected::apply(autograd::ComputationPtr input) {
  if (input->dim() != _kernel->getInputDim()) {
    std::stringstream error;
    error << "Cannot apply FullyConnected op with weight matrix of shape ("
          << _kernel->getDim() << ", " << _kernel->getInputDim()
          << ") to input tensor with dim " << input->dim() << ".";

    throw std::invalid_argument(error.str());
  }
  return autograd::Computation::make(shared_from_this(), {std::move(input)});
}

uint32_t FullyConnected::inputDim() const { return _kernel->getInputDim(); }

const float* FullyConnected::weightsPtr() const {
  return _kernel->getWeightsPtr();
}

const float* FullyConnected::biasesPtr() const {
  return _kernel->getBiasesPtr();
}

void FullyConnected::freezeHashTables(bool insert_labels_if_not_found) {
  _kernel->freezeHashTables(insert_labels_if_not_found);
}

void FullyConnected::setWeightsAndBiases(const float* weights,
                                         const float* biases) {
  _kernel->setWeights(weights);
  _kernel->setBiases(biases);
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

  _kernel->initOptimizer();
}

}  // namespace thirdai::bolt::nn::ops

namespace cereal {

/**
 * This is because the Op base class only uses a serialize function, whereas
 * this Op uses a load/save pair. This tells cereal to use the load save pair
 * instead of the serialize method of the parent class. See docs here:
 * https://uscilab.github.io/cereal/serialization_functions.html#inheritance
 */
template <class Archive>
struct specialize<Archive, thirdai::bolt::nn::ops::FullyConnected,
                  cereal::specialization::member_load_save> {};

}  // namespace cereal

CEREAL_REGISTER_TYPE(thirdai::bolt::nn::ops::FullyConnected)