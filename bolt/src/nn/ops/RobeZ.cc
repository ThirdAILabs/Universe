#include "RobeZ.h"
#include <cereal/archives/binary.hpp>
#include <cereal/specialize.hpp>
#include <cereal/types/base_class.hpp>
#include <cereal/types/memory.hpp>
#include <bolt/src/nn/autograd/Computation.h>
#include <bolt/src/nn/ops/Op.h>
#include <bolt/src/utils/ProtobufUtils.h>

namespace thirdai::bolt::nn::ops {

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

void RobeZ::forward(const autograd::ComputationList& inputs,
                    tensor::TensorPtr& output, uint32_t index_in_batch,
                    bool training) {
  (void)training;
  assert(inputs.size() == 1);

  _kernel->forward(inputs[0]->tensor()->getVector(index_in_batch),
                   output->getVector(index_in_batch));
}

void RobeZ::backpropagate(autograd::ComputationList& inputs,
                          tensor::TensorPtr& output, uint32_t index_in_batch) {
  assert(inputs.size() == 1);

  _kernel->backpropagate(inputs[0]->tensor()->getVector(index_in_batch),
                         output->getVector(index_in_batch));
}

void RobeZ::updateParameters(float learning_rate, uint32_t train_steps) {
  _kernel->updateParameters(learning_rate, train_steps, BETA1, BETA2, EPS);
}

uint32_t RobeZ::dim() const { return _kernel->getOutputDim(); }

std::optional<uint32_t> RobeZ::nonzeros(const autograd::ComputationList& inputs,
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

bolt_proto::Op RobeZ::toProto(bool with_optimizer) const {
  bolt_proto::Op op;
  op.set_name(name());

  auto* robez = op.mutable_robez();

  robez->set_num_lookups_per_token(_kernel->numEmbeddingLookups());
  robez->set_lookup_size(_kernel->lookupSize());
  robez->set_log_embedding_block_size(_kernel->logEmbeddingBlockSize());

  switch (_kernel->reductionType()) {
    case EmbeddingReductionType::CONCATENATION:
      robez->set_reduction(bolt_proto::EmbeddingReduction::CONCAT);
      break;
    case EmbeddingReductionType::SUM:
      robez->set_reduction(bolt_proto::EmbeddingReduction::SUM);
      break;
    case EmbeddingReductionType::AVERAGE:
      robez->set_reduction(bolt_proto::EmbeddingReduction::AVG);
      break;
  }

  if (_kernel->numTokensPerInput()) {
    robez->set_num_tokens_per_input(*_kernel->numTokensPerInput());
  }
  robez->set_update_chunk_size(_kernel->updateChunkSize());
  robez->set_hash_seed(_kernel->hashSeed());

  robez->set_allocated_embedding_block(
      utils::parametersToProto(_kernel->getRawEmbeddingBlock()));

  const auto& optimizer = _kernel->optimizer();
  if (with_optimizer && optimizer) {
    robez->set_allocated_embedding_block_optimizer(utils::optimizerToProto(
        *optimizer, _kernel->numEmbeddingChunks(), _kernel->updateChunkSize()));
  }

  return op;
}

void RobeZ::summary(std::ostream& summary,
                    const autograd::ComputationList& inputs,
                    const autograd::Computation* output) const {
  summary << "RobeZ(" << name() << "): " << inputs[0]->name() << " -> "
          << output->name() << " [";
  _kernel->buildLayerSummary(summary);
  summary << "]";
}

void RobeZ::setSerializeOptimizer(bool should_serialize_optimizer) {
  _kernel->saveWithOptimizer(should_serialize_optimizer);
}

autograd::ComputationPtr RobeZ::apply(autograd::ComputationPtr input) {
  return autograd::Computation::make(shared_from_this(), {std::move(input)});
}

template void RobeZ::save(cereal::BinaryOutputArchive&) const;

template <class Archive>
void RobeZ::save(Archive& archive) const {
  archive(cereal::base_class<Op>(this), _kernel);
}

template void RobeZ::load(cereal::BinaryInputArchive&);

template <class Archive>
void RobeZ::load(Archive& archive) {
  archive(cereal::base_class<Op>(this), _kernel);

  _kernel->initOptimizer();
}

std::shared_ptr<RobeZ> RobeZ::duplicateWithNewReduction(
    const std::string& reduction,
    std::optional<uint64_t> num_tokens_per_input) {
  auto new_kernel =
      _kernel->duplicateWithNewReduction(reduction, num_tokens_per_input);

  std::string new_name = nextRobeZOpName() + "_shared_" + name();
  return std::shared_ptr<RobeZ>(new RobeZ(std::move(new_kernel), new_name));
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
struct specialize<Archive, thirdai::bolt::nn::ops::RobeZ,
                  cereal::specialization::member_load_save> {};

}  // namespace cereal

CEREAL_REGISTER_TYPE(thirdai::bolt::nn::ops::RobeZ)