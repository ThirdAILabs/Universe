#include "PatchEmbedding.h"
#include <bolt/src/layers/FullyConnectedLayer.h>
#include <bolt/src/nn/autograd/Computation.h>
#include <bolt_vector/src/BoltVector.h>
#include <archive/src/Archive.h>
#include <archive/src/Map.h>
#include <archive/src/ParameterReference.h>
#include <stdexcept>

namespace thirdai::bolt {

std::string nextPatchEmbeddingOpName() {
  static uint32_t constructed = 0;
  return "patch_emb_" + std::to_string(++constructed);
}

PatchEmbedding::PatchEmbedding(size_t emb_dim, size_t patch_dim,
                               size_t n_patches, float sparsity,
                               const std::string& activation,
                               SamplingConfigPtr sampling, bool use_bias,
                               size_t rebuild_hash_tables,
                               size_t reconstruct_hash_functions)
    : Op(nextPatchEmbeddingOpName()),
      _n_patches(n_patches),
      _rebuild_hash_tables(rebuild_hash_tables),
      _reconstruct_hash_functions(reconstruct_hash_functions),
      _updates_since_rebuild_hash_tables(0),
      _updates_since_reconstruct_hash_functions(0) {
  if (!sampling) {
    sampling = DWTASamplingConfig::autotune(emb_dim, sparsity,
                                            /* experimental_autotune=*/false);
  }
  FullyConnectedLayerConfig config(emb_dim, sparsity, activation,
                                   std::move(sampling));

  _kernel = std::make_unique<FullyConnectedLayer>(
      config, patch_dim, /* disable_sparse_sparse_updates */ false, use_bias);
}

void PatchEmbedding::forward(const ComputationList& inputs, TensorPtr& output,
                             uint32_t index_in_batch, bool training) {
  (void)training;
  assert(inputs.size() == 1);

  const BoltVector& input = inputs.at(0)->tensor()->getVector(index_in_batch);
  if (!input.isDense()) {
    throw std::invalid_argument("Expected dense patches.");
  }

  const BoltVector& output_vec = output->getVector(index_in_batch);

  size_t patch_dim = patchDim();
  size_t emb_nonzeros = patchNonzeros(/*use_sparsity=*/output_vec.len < dim());

  for (size_t i = 0; i < _n_patches; i++) {
    BoltVector patch = input.viewChunk(i, patch_dim);
    BoltVector patch_emb = output_vec.viewChunk(i, emb_nonzeros);

    _kernel->forward(patch, patch_emb, nullptr);
  }
}

void PatchEmbedding::backpropagate(ComputationList& inputs, TensorPtr& output,
                                   uint32_t index_in_batch) {
  assert(inputs.size() == 1);

  const BoltVector& input = inputs.at(0)->tensor()->getVector(index_in_batch);
  if (!input.isDense()) {
    throw std::invalid_argument("Expected dense patches.");
  }

  const BoltVector& output_vec = output->getVector(index_in_batch);

  size_t patch_dim = patchDim();
  size_t emb_nonzeros = patchNonzeros(/*use_sparsity=*/output_vec.len < dim());

  for (size_t i = 0; i < _n_patches; i++) {
    BoltVector patch = input.viewChunk(i, patch_dim);
    BoltVector patch_emb = output_vec.viewChunk(i, emb_nonzeros);

    if (patch.hasGradients()) {
      _kernel->backpropagate(patch, patch_emb);
    } else {
      _kernel->backpropagateInputLayer(patch, patch_emb);
    }
  }
}

void PatchEmbedding::updateParameters(float learning_rate,
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

uint32_t PatchEmbedding::dim() const { return _kernel->getDim() * _n_patches; }

std::optional<uint32_t> PatchEmbedding::nonzeros(const ComputationList& inputs,
                                                 bool use_sparsity) const {
  // The number of output nonzeros for a FullyConnected op do not depend on its
  // inputs.
  (void)inputs;

  return patchNonzeros(use_sparsity) * _n_patches;
}

void PatchEmbedding::initOptimizer() { _kernel->initOptimizer(); }

void PatchEmbedding::disableSparseParameterUpdates() {
  _kernel->disableSparseParameterUpdates();
}

void PatchEmbedding::enableSparseParameterUpdates() {
  _kernel->enableSparseParameterUpdates();
}

std::vector<std::vector<float>*> PatchEmbedding::gradients() {
  return {&_kernel->weightsGradient(), &_kernel->biasGradient()};
}

std::vector<std::vector<float>*> PatchEmbedding::parameters() {
  return {&_kernel->weights(), &_kernel->biases()};
}

ComputationPtr PatchEmbedding::applyToInputs(const ComputationList& inputs) {
  if (inputs.size() != 2) {
    throw std::invalid_argument(
        "Expected PatchEmbedding op to have one input.");
  }
  return apply(inputs.at(0));
}

ar::ConstArchivePtr PatchEmbedding::toArchive(bool with_optimizer) const {
  (void)with_optimizer;

  auto map = baseArchive();
  map->set("type", ar::str(type()));

  map->set("n_patches", ar::u64(_n_patches));
  map->set("dim", ar::u64(patchEmbeddingDim()));
  map->set("input_dim", ar::u64(patchDim()));
  map->set("sparsity", ar::f32(_kernel->_sparsity));
  map->set("activation", ar::str(activationFunctionToStr(_kernel->_act_func)));
  map->set("use_bias", ar::boolean(_kernel->_use_bias));

  map->set("weights",
           ar::ParameterReference::make(_kernel->_weights, shared_from_this()));
  map->set("biases",
           ar::ParameterReference::make(_kernel->_biases, shared_from_this()));

  if (auto neuron_index = _kernel->neuronIndex()) {
    map->set("neuron_index", neuron_index->toArchive());
  }
  map->set("index_frozen", ar::boolean(_kernel->_index_frozen));
  map->set("rebuild_hash_tables", ar::u64(_rebuild_hash_tables));
  map->set("reconstruct_hash_functions", ar::u64(_reconstruct_hash_functions));

  if (with_optimizer && _kernel->_weight_optimizer &&
      _kernel->_bias_optimizer) {
    map->set("weight_optimizer",
             optimizerToArchive(*_kernel->_weight_optimizer, shared_from_this(),
                                patchEmbeddingDim(), patchDim()));

    map->set("bias_optimizer",
             optimizerToArchive(*_kernel->_bias_optimizer, shared_from_this(),
                                /*rows=*/1, patchEmbeddingDim()));
  }

  map->set("disable_sparse_parameter_updates",
           ar::boolean(_kernel->_disable_sparse_parameter_updates));

  return map;
}

std::shared_ptr<PatchEmbedding> PatchEmbedding::fromArchive(
    const ar::Archive& archive) {
  return std::shared_ptr<PatchEmbedding>(new PatchEmbedding(archive));
}

PatchEmbedding::PatchEmbedding(const ar::Archive& archive)
    : Op(archive.str("name")),
      _kernel(std::make_unique<FullyConnectedLayer>(archive)),
      _n_patches(archive.u64("n_patches")),
      _rebuild_hash_tables(archive.u64("rebuild_hash_tables")),
      _reconstruct_hash_functions(archive.u64("reconstruct_hash_functions")),
      _updates_since_rebuild_hash_tables(0),
      _updates_since_reconstruct_hash_functions(0) {
  assertOpType(archive, type());
}

void PatchEmbedding::summary(std::ostream& summary,
                             const ComputationList& inputs,
                             const Computation* output) const {
  summary << "PatchEmbedding(" << name() << "): " << inputs[0]->name() << " -> "
          << output->name();
  summary << " [dim=" << _kernel->getDim()
          << ", sparsity=" << _kernel->getSparsity() << ", activation="
          << activationFunctionToStr(_kernel->getActivationFunction())
          << ", n_patches=" << _n_patches;
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

void PatchEmbedding::setSerializeOptimizer(bool should_serialize_optimizer) {
  _kernel->saveWithOptimizer(should_serialize_optimizer);
}

ComputationPtr PatchEmbedding::apply(ComputationPtr input) {
  if (input->dim() != _kernel->getInputDim() * _n_patches) {
    std::stringstream error;
    error << "Cannot apply PatchEmbedding op with weight matrix of shape ("
          << _kernel->getDim() << ", " << _kernel->getInputDim()
          << ") and n_patches=" + std::to_string(_n_patches) +
                 " to input tensor with dim "
          << input->dim() << ".";

    throw std::invalid_argument(error.str());
  }
  return Computation::make(shared_from_this(), {std::move(input)});
}

void PatchEmbedding::setWeights(const float* weights) {
  _kernel->setWeights(weights);
}

void PatchEmbedding::setBiases(const float* new_biases) {
  _kernel->setBiases(new_biases);
}

void PatchEmbedding::setHashTable(hashing::HashFunctionPtr hash_fn,
                                  hashtable::SampledHashTablePtr hash_table) {
  return _kernel->setHashTable(std::move(hash_fn), std::move(hash_table));
}

uint32_t PatchEmbedding::patchEmbeddingDim() const { return _kernel->getDim(); }

uint32_t PatchEmbedding::patchDim() const { return _kernel->getInputDim(); }

size_t PatchEmbedding::patchNonzeros(bool use_sparsity) const {
  if (use_sparsity) {
    return _kernel->getSparseDim();
  }
  return _kernel->getDim();
}

}  // namespace thirdai::bolt