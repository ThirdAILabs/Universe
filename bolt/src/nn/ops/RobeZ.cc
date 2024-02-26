#include "RobeZ.h"
#include <cereal/archives/binary.hpp>
#include <cereal/specialize.hpp>
#include <cereal/types/base_class.hpp>
#include <cereal/types/memory.hpp>
#include <bolt/src/layers/EmbeddingLayer.h>
#include <bolt/src/nn/autograd/Computation.h>
#include <bolt/src/nn/ops/Op.h>
#include <archive/src/Archive.h>
#include <archive/src/Map.h>
#include <archive/src/ParameterReference.h>
#include <memory>

namespace thirdai::bolt {

std::string nextRobeZOpName() {
  static uint32_t constructed = 0;
  return "robez_" + std::to_string(++constructed);
}

RobeZ::RobeZ(uint64_t num_embedding_lookups, uint64_t lookup_size,
             uint64_t log_embedding_block_size, const std::string& reduction,
             std::optional<uint64_t> num_tokens_per_input,
             uint64_t update_chunk_size, uint32_t seed)
    : Op(nextRobeZOpName()) {
  EmbeddingLayerConfig config(
      /* num_embedding_lookups= */ num_embedding_lookups,
      /* lookup_size= */ lookup_size,
      /* log_embedding_block_size= */ log_embedding_block_size,
      /* update_chunk_size= */ update_chunk_size, /* reduction= */ reduction,
      /* num_tokens_per_input= */ num_tokens_per_input);

  _kernel = std::make_unique<EmbeddingLayer>(config, seed);
}

std::shared_ptr<RobeZ> RobeZ::make(uint64_t num_embedding_lookups,
                                   uint64_t lookup_size,
                                   uint64_t log_embedding_block_size,
                                   const std::string& reduction,
                                   std::optional<uint64_t> num_tokens_per_input,
                                   uint64_t update_chunk_size, uint32_t seed) {
  return std::shared_ptr<RobeZ>(new RobeZ(
      /* num_embedding_lookups= */ num_embedding_lookups,
      /* lookup_size= */ lookup_size,
      /* log_embedding_block_size= */ log_embedding_block_size,
      /* reduction= */ reduction,
      /* num_tokens_per_input= */ num_tokens_per_input,
      /* update_chunk_size= */ update_chunk_size, seed));
}

void RobeZ::forward(const ComputationList& inputs, TensorPtr& output,
                    uint32_t index_in_batch, bool training) {
  (void)training;
  assert(inputs.size() == 1);

  _kernel->forward(inputs[0]->tensor()->getVector(index_in_batch),
                   output->getVector(index_in_batch));
}

void RobeZ::backpropagate(ComputationList& inputs, TensorPtr& output,
                          uint32_t index_in_batch) {
  assert(inputs.size() == 1);

  _kernel->backpropagate(inputs[0]->tensor()->getVector(index_in_batch),
                         output->getVector(index_in_batch));
}

void RobeZ::updateParameters(float learning_rate, uint32_t train_steps) {
  _kernel->updateParameters(learning_rate, train_steps);
}

void RobeZ::initOptimizer(const OptimizerFactoryPtr& optimizer_factory,
                          bool replace_existing_optimizer) {
  _kernel->initOptimizer(optimizer_factory, replace_existing_optimizer);
}

uint32_t RobeZ::dim() const { return _kernel->getOutputDim(); }

std::optional<uint32_t> RobeZ::nonzeros(const ComputationList& inputs,
                                        bool use_sparsity) const {
  (void)inputs;
  (void)use_sparsity;

  return dim();
}

void RobeZ::disableSparseParameterUpdates() {
  _kernel->disableSparseParameterUpdates();
}

void RobeZ::enableSparseParameterUpdates() {
  _kernel->enableSparseParameterUpdates();
}

std::vector<std::vector<float>*> RobeZ::gradients() {
  return {&_kernel->getRawEmbeddingBlockGradient()};
}

std::vector<std::vector<float>*> RobeZ::parameters() {
  return {&_kernel->getRawEmbeddingBlock()};
}

ComputationPtr RobeZ::applyToInputs(const ComputationList& inputs) {
  if (inputs.size() != 1) {
    throw std::invalid_argument("Expected RobeZ op to have one input.");
  }
  return apply(inputs.at(0));
}

ar::ConstArchivePtr RobeZ::toArchive(bool with_optimizer) const {
  (void)with_optimizer;

  auto map = baseArchive();
  map->set("type", ar::str(type()));

  map->set("num_lookups_per_token", ar::u64(_kernel->_num_lookups_per_token));
  map->set("lookup_size", ar::u64(_kernel->_lookup_size));
  map->set("log_embedding_block_size",
           ar::u64(_kernel->_log_embedding_block_size));
  map->set("reduction", ar::str(reductionToString(_kernel->_reduction)));
  if (_kernel->_num_tokens_per_input) {
    map->set("num_tokens_per_input", ar::u64(*_kernel->_num_tokens_per_input));
  }
  map->set("update_chunk_size", ar::u64(_kernel->_update_chunk_size));
  map->set("hash_seed", ar::u64(_kernel->_hash_fn.seed()));

  map->set("embeddings", ar::ParameterReference::make(
                             *_kernel->_embedding_block, shared_from_this()));

  if (with_optimizer && _kernel->_optimizer) {
    map->set("embedding_optimizer",
             _kernel->_optimizer->toArchive(shared_from_this()));
  }

  map->set("disable_sparse_parameter_updates",
           ar::boolean(_kernel->_disable_sparse_parameter_updates));

  return map;
}

std::shared_ptr<RobeZ> RobeZ::fromArchive(const ar::Archive& archive) {
  return std::shared_ptr<RobeZ>(new RobeZ(archive));
}

RobeZ::RobeZ(const ar::Archive& archive)
    : Op(archive.str("name")),
      _kernel(std::make_unique<EmbeddingLayer>(archive)) {
  assertOpType(archive, type());
}

void RobeZ::summary(std::ostream& summary, const ComputationList& inputs,
                    const Computation* output) const {
  summary << "RobeZ(" << name() << "): " << inputs[0]->name() << " -> "
          << output->name() << " [";
  _kernel->buildLayerSummary(summary);
  summary << "]";
}

void RobeZ::setSerializeOptimizer(bool should_serialize_optimizer) {
  _kernel->saveWithOptimizer(should_serialize_optimizer);
}

std::vector<std::pair<std::string, double>> RobeZ::parameterAndGradNorms()
    const {
  std::vector<std::pair<std::string, double>> all_norms;

  computeNorms(_kernel->getRawEmbeddingBlock(), "embeddings", all_norms);
  if (_kernel->hasOptimizer()) {
    computeNorms(_kernel->getRawEmbeddingBlockGradient(), "embeddings_grad",
                 all_norms);
  }

  return all_norms;
}

ComputationPtr RobeZ::apply(ComputationPtr input) {
  return Computation::make(shared_from_this(), {std::move(input)});
}

template void RobeZ::serialize(cereal::BinaryInputArchive&);
template void RobeZ::serialize(cereal::BinaryOutputArchive&);

template <class Archive>
void RobeZ::serialize(Archive& archive) {
  archive(cereal::base_class<Op>(this), _kernel);
}

std::shared_ptr<RobeZ> RobeZ::duplicateWithNewReduction(
    const std::string& reduction,
    std::optional<uint64_t> num_tokens_per_input) {
  auto new_kernel =
      _kernel->duplicateWithNewReduction(reduction, num_tokens_per_input);

  std::string new_name = nextRobeZOpName() + "_shared_" + name();
  return std::shared_ptr<RobeZ>(new RobeZ(std::move(new_kernel), new_name));
}

}  // namespace thirdai::bolt

CEREAL_REGISTER_TYPE_WITH_NAME(thirdai::bolt::RobeZ,
                               "thirdai::bolt::nn::ops::RobeZ")
