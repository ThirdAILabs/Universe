#include "FCMixer.h"
#include <cereal/archives/binary.hpp>
#include <cereal/specialize.hpp>
#include <cereal/types/base_class.hpp>
#include <cereal/types/memory.hpp>
#include <cereal/types/polymorphic.hpp>
#include <bolt/src/layers/LayerUtils.h>
#include <bolt/src/nn/model/Model.h>
#include <bolt/src/nn/ops/Op.h>
#include <bolt/src/nn/tensor/Tensor.h>
#include <bolt_vector/src/BoltVector.h>
#include <bolt_vector/src/BoltVectorUtils.h>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <utility>

namespace thirdai::bolt::nn::ops {

std::string nextFCMixerOpName() {
  static uint32_t constructed = 0;
  return "mixer_" + std::to_string(++constructed);
}
/**
 * dim = the output dimension of the FC Kernel.
 * input_dim = dim of the input to the FC Kernel.
 * number_segment = the number of rows in the matrix viz of the input.

 * The input has the dimension input_dim and we assume that the input
 tensor/boltvector is in row major format. If we visualize the input as a
 matrix, the number of rows in the matrix would be equal to the number of
 segment and the number of columns would be equal to the output dimension of the
 FC Kernel.

 * This operation operates on a single row of the input. That is, the input is
 split into number_segment number of parts and FC kernel is applied on each of
 the input segments and then finally the segments are concatenated. We can think
 of this layer as a FCMixer over the rows. That is, it mixes the values inside
 of a row.
 */
FCMixer::FCMixer(uint32_t dim, uint32_t input_dim, uint32_t number_segment,
                 float sparsity, const std::string& activation,
                 SamplingConfigPtr sampling, bool use_bias,
                 uint32_t rebuild_hash_tables,
                 uint32_t reconstruct_hash_functions)
    : Op(nextFCMixerOpName()),
      _rows(number_segment),
      _columns(dim),
      _rebuild_hash_tables(rebuild_hash_tables),
      _reconstruct_hash_functions(reconstruct_hash_functions),
      _updates_since_rebuild_hash_tables(0),
      _updates_since_reconstruct_hash_functions(0) {
  if (sparsity < 1) {
    throw std::logic_error("FCMixer with sparse kernel not implemented.");
  }
  if (_rows * _columns != input_dim) {
    throw std::logic_error(
        "FCMixer have the number of rows*columns = input_dim.");
  }
  if (!sampling) {
    sampling = DWTASamplingConfig::autotune(dim, sparsity,
                                            /* experimental_autotune=*/false);
  }
  FullyConnectedLayerConfig config(dim, sparsity, activation,
                                   std::move(sampling));

  _kernel = std::make_shared<FullyConnectedLayer>(
      config, dim, /* disable_sparse_sparse_updates */ false, use_bias);
}

std::shared_ptr<FCMixer> FCMixer::make(
    uint32_t dim, uint32_t input_dim, uint32_t number_segment, float sparsity,
    const std::string& activation, SamplingConfigPtr sampling, bool use_bias,
    uint32_t rebuild_hash_tables, uint32_t reconstruct_hash_functions) {
  return std::shared_ptr<FCMixer>(new FCMixer(
      dim, input_dim, number_segment, sparsity, activation, std::move(sampling),
      use_bias, rebuild_hash_tables, reconstruct_hash_functions));
}

void FCMixer::forward(const autograd::ComputationList& inputs,
                      tensor::TensorPtr& output, uint32_t index_in_batch,
                      bool training) {
  assert(inputs.size() == 1 || inputs.size() == 2);
  // If the op is an output pass in labels during training to ensure labels are
  // in active neuron set.
  const BoltVector* labels = nullptr;
  if (training && inputs.size() == 2) {
    labels = &inputs[1]->tensor()->getVector(index_in_batch);
  }
  if (labels != nullptr) {
    throw std::logic_error("FCMixers should not have non null label pointers.");
  }

  std::vector<BoltVector> segmented_row_vector_output =
      bolt_vector::segmentRowMajorVector(output->getVector(index_in_batch),
                                         _rows, _columns);
  if (segmented_row_vector_output.size() != _rows) {
    throw std::logic_error(
        "The size of the segmented output bolt vector should be equal to the "
        "number of segments");
  }

  std::vector<BoltVector> segmented_row_vector_input =
      bolt_vector::segmentRowMajorVector(
          inputs[0]->tensor()->getVector(index_in_batch), _rows, _columns);
  if (segmented_row_vector_input.size() != _rows) {
    throw std::logic_error(
        "The size of the segmented input bolt vector should be equal to the "
        "number of segments");
  }
  for (size_t i = 0; i < segmented_row_vector_input.size(); i++) {
    _kernel->forward(segmented_row_vector_input[i],
                     segmented_row_vector_output[i], labels);
  }
}

void FCMixer::backpropagate(autograd::ComputationList& inputs,
                            tensor::TensorPtr& output,
                            uint32_t index_in_batch) {
  assert(inputs.size() == 1 || inputs.size() == 2);

  BoltVector& input = inputs[0]->tensor()->getVector(index_in_batch);

  std::vector<BoltVector> segmented_row_vector_output =
      bolt_vector::segmentRowMajorVector(output->getVector(index_in_batch),
                                         _rows, _columns);
  std::vector<BoltVector> segmented_row_vector_input =
      bolt_vector::segmentRowMajorVector(
          inputs[0]->tensor()->getVector(index_in_batch), _rows, _columns);

  for (size_t i = 0; i < segmented_row_vector_input.size(); i++) {
    if (input.hasGradients()) {
      _kernel->backpropagate(segmented_row_vector_input[i],
                             segmented_row_vector_output[i]);
    } else {
      _kernel->backpropagateInputLayer(segmented_row_vector_input[i],
                                       segmented_row_vector_output[i]);
    }
  }
}

void FCMixer::updateParameters(float learning_rate, uint32_t train_steps) {
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

uint32_t FCMixer::dim() const { return _rows * _columns; }

std::optional<uint32_t> FCMixer::nonzeros(
    const autograd::ComputationList& inputs, bool use_sparsity) const {
  // The number of output nonzeros for a FullyConnected op do not depend on its
  // inputs.
  (void)inputs;
  (void)use_sparsity;

  return _rows * _columns;
}

void FCMixer::disableSparseParameterUpdates() {
  _kernel->disableSparseParameterUpdates();
}

void FCMixer::enableSparseParameterUpdates() {
  _kernel->enableSparseParameterUpdates();
}

std::vector<std::vector<float>*> FCMixer::gradients() {
  return {&_kernel->weightsGradient(), &_kernel->biasGradient()};
}

std::vector<std::vector<float>*> FCMixer::parameters() {
  return {&_kernel->weights(), &_kernel->biases()};
}

void FCMixer::summary(std::ostream& summary,
                      const autograd::ComputationList& inputs,
                      const autograd::Computation* output) const {
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

  summary << ", rows=" << _rows;
  summary << ", columns=" << _columns;
  summary << "]";
}

void FCMixer::setSerializeOptimizer(bool should_serialize_optimizer) {
  _kernel->saveWithOptimizer(should_serialize_optimizer);
}

void FCMixer::reBuildHashFunction() { _kernel->reBuildHashFunction(); }
void FCMixer::registerModel(const std::weak_ptr<model::Model>& new_model) {
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

autograd::ComputationPtr FCMixer::apply(autograd::ComputationPtr input) {
  if (input->dim() != _rows * _columns) {
    std::stringstream error;
    error << "Cannot apply FullyConnected op with weight matrix of shape ("
          << _kernel->getDim() << ", " << _kernel->getInputDim()
          << ") to input tensor with dim " << input->dim() << ".";

    throw std::invalid_argument(error.str());
  }
  return autograd::Computation::make(shared_from_this(), {std::move(input)});
}

uint32_t FCMixer::inputDim() const { return _rows * _columns; }

const float* FCMixer::weightsPtr() const { return _kernel->getWeightsPtr(); }

const float* FCMixer::biasesPtr() const { return _kernel->getBiasesPtr(); }

std::shared_ptr<FullyConnectedLayer> FCMixer::kernel() const { return _kernel; }

void FCMixer::freezeHashTables(bool insert_labels_if_not_found) {
  _kernel->freezeHashTables(insert_labels_if_not_found);
}

void FCMixer::unfreezeHashTables() { _kernel->unfreezeHashTables(); }

void FCMixer::setWeights(const float* weights) { _kernel->setWeights(weights); }

void FCMixer::setBiases(const float* new_biases) {
  _kernel->setBiases(new_biases);
}

std::pair<hashing::HashFunctionPtr, hashtable::SampledHashTablePtr>
FCMixer::getHashTable() const {
  return _kernel->getHashTable();
}

void FCMixer::setHashTable(hashing::HashFunctionPtr hash_fn,
                           hashtable::SampledHashTablePtr hash_table) {
  return _kernel->setHashTable(std::move(hash_fn), std::move(hash_table));
}

void FCMixer::autotuneRehashRebuild(uint32_t num_batches, uint32_t batch_size) {
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

void FCMixer::setSparsity(float sparsity, bool rebuild_hash_tables,
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

template void FCMixer::save(cereal::BinaryOutputArchive&) const;

template <class Archive>
void FCMixer::save(Archive& archive) const {
  archive(cereal::base_class<Op>(this), _kernel, _rows, _columns,
          _rebuild_hash_tables, _reconstruct_hash_functions,
          _updates_since_rebuild_hash_tables,
          _updates_since_reconstruct_hash_functions);
}

template void FCMixer::load(cereal::BinaryInputArchive&);

template <class Archive>
void FCMixer::load(Archive& archive) {
  archive(cereal::base_class<Op>(this), _kernel, _rows, _columns,
          _rebuild_hash_tables, _reconstruct_hash_functions,
          _updates_since_rebuild_hash_tables,
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
struct specialize<Archive, thirdai::bolt::nn::ops::FCMixer,
                  cereal::specialization::member_load_save> {};

}  // namespace cereal

CEREAL_REGISTER_TYPE(thirdai::bolt::nn::ops::FCMixer)