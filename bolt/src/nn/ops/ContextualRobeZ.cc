#include "ContextualRobeZ.h"
#include <cereal/archives/binary.hpp>
#include <cereal/specialize.hpp>
#include <cereal/types/base_class.hpp>
#include <cereal/types/memory.hpp>
#include <bolt/src/nn/autograd/Computation.h>
#include <bolt/src/nn/ops/Op.h>
#include <bolt_vector/src/BoltVector.h>
#include <bolt_vector/src/BoltVectorUtils.h>
namespace thirdai::bolt::nn::ops {

std::string nextContextualRobeZOpName() {
  static uint32_t constructed = 0;
  return "ContextualRobeZ_" + std::to_string(++constructed);
}

ContextualRobeZ::ContextualRobeZ(uint64_t num_embedding_lookups,
                                 uint64_t lookup_size,
                                 uint64_t log_embedding_block_size,
                                 uint64_t num_tokens_per_input,
                                 uint64_t update_chunk_size)
    : Op(nextContextualRobeZOpName()) {
  EmbeddingLayerConfig config(
      /* num_embedding_lookups= */ num_embedding_lookups,
      /* lookup_size= */ lookup_size,
      /* log_embedding_block_size= */ log_embedding_block_size,
      /* update_chunk_size= */ update_chunk_size, /* reduction= */ "concat",
      /* num_tokens_per_input= */ num_tokens_per_input);

  _kernel = std::make_unique<EmbeddingLayer>(config);
}

std::shared_ptr<ContextualRobeZ> ContextualRobeZ::make(
    uint64_t num_embedding_lookups, uint64_t lookup_size,
    uint64_t log_embedding_block_size, uint64_t num_tokens_per_input,
    uint64_t update_chunk_size) {
  return std::shared_ptr<ContextualRobeZ>(new ContextualRobeZ(
      /* num_embedding_lookups= */ num_embedding_lookups,
      /* lookup_size= */ lookup_size,
      /* log_embedding_block_size= */ log_embedding_block_size,
      /* num_tokens_per_input= */ num_tokens_per_input,
      /* update_chunk_size= */ update_chunk_size));
}

void ContextualRobeZ::forward(const autograd::ComputationList& inputs,
                              tensor::TensorPtr& output,
                              uint32_t index_in_batch, bool training) {
  (void)training;
  assert(inputs.size() == 1);
  BoltVector output_vector = output->getVector(index_in_batch);
  BoltVector concatenated_output =
      thirdai::bolt_vector::getBoltVectorWithOffset(
          output->getVector(index_in_batch), _kernel->getOutputDim(), 0);
  _kernel->forward(inputs[0]->tensor()->getVector(index_in_batch),
                   concatenated_output);

  uint64_t number_tokens = _kernel->getNumberTokens();
  uint64_t embedding_size =
      _kernel->getLookupSize() * _kernel->getNumberLookups();
  for (uint64_t token_index = 0; token_index < number_tokens; token_index++) {
    for (uint64_t embedding_index = 0; embedding_index < embedding_size;
         embedding_index++) {
      output_vector.activations[_kernel->getOutputDim() + embedding_index] +=
          concatenated_output
              .activations[token_index * embedding_size + embedding_index];
    }
  }
  for (uint64_t embedding_index = 0; embedding_index < embedding_size;
       embedding_index++) {
    output_vector.activations[_kernel->getOutputDim() + embedding_index] /=
        number_tokens;
  }
}

void ContextualRobeZ::backpropagate(autograd::ComputationList& inputs,
                                    tensor::TensorPtr& output,
                                    uint32_t index_in_batch) {
  assert(inputs.size() == 1);
  uint64_t number_tokens = _kernel->getNumberTokens();
  uint64_t embedding_size =
      _kernel->getLookupSize() * _kernel->getNumberLookups();
  BoltVector output_vector = output->getVector(index_in_batch);

  BoltVector concatenated_output =
      thirdai::bolt_vector::getBoltVectorWithOffset(
          output->getVector(index_in_batch), _kernel->getOutputDim(), 0);

  for (uint64_t token_index = 0; token_index < number_tokens; token_index++) {
    for (uint64_t embedding_index = 0; embedding_index < embedding_size;
         embedding_index++) {
      concatenated_output
          .activations[token_index * embedding_size + embedding_index] +=
          output_vector.gradients[_kernel->getOutputDim() + embedding_index] /
          number_tokens;
    }
  }
  _kernel->backpropagate(inputs[0]->tensor()->getVector(index_in_batch),
                         concatenated_output);
}

void ContextualRobeZ::updateParameters(float learning_rate,
                                       uint32_t train_steps) {
  _kernel->updateParameters(learning_rate, train_steps, BETA1, BETA2, EPS);
}

uint32_t ContextualRobeZ::dim() const {
  return _kernel->getOutputDim() +
         _kernel->getLookupSize() * _kernel->getNumberLookups();
}

std::optional<uint32_t> ContextualRobeZ::nonzeros(
    const autograd::ComputationList& inputs, bool use_sparsity) const {
  (void)inputs;
  (void)use_sparsity;

  return dim();
}

void ContextualRobeZ::disableSparseParameterUpdates() {
  _kernel->disableSparseParameterUpdates();
}

std::vector<std::vector<float>*> ContextualRobeZ::gradients() {
  return {&_kernel->getRawEmbeddingBlockGradient()};
}

std::vector<std::vector<float>*> ContextualRobeZ::parameters() {
  return {&_kernel->getRawEmbeddingBlock()};
}

void ContextualRobeZ::summary(std::ostream& summary,
                              const autograd::ComputationList& inputs,
                              const autograd::Computation* output) const {
  summary << "ContextualRobeZ(" << name() << "): " << inputs[0]->name()
          << " -> " << output->name() << " [";
  _kernel->buildLayerSummary(summary);
  summary << "]";
}

void ContextualRobeZ::setSerializeOptimizer(bool should_serialize_optimizer) {
  _kernel->saveWithOptimizer(should_serialize_optimizer);
}

autograd::ComputationPtr ContextualRobeZ::apply(
    autograd::ComputationPtr input) {
  return autograd::Computation::make(shared_from_this(), {std::move(input)});
}

template void ContextualRobeZ::save(cereal::BinaryOutputArchive&) const;

template <class Archive>
void ContextualRobeZ::save(Archive& archive) const {
  archive(cereal::base_class<Op>(this), _kernel);
}

template void ContextualRobeZ::load(cereal::BinaryInputArchive&);

template <class Archive>
void ContextualRobeZ::load(Archive& archive) {
  archive(cereal::base_class<Op>(this), _kernel);

  _kernel->initOptimizer();
}

std::shared_ptr<ContextualRobeZ> ContextualRobeZ::duplicateWithNewReduction(
    const std::string& reduction,
    std::optional<uint64_t> num_tokens_per_input) {
  auto new_kernel =
      _kernel->duplicateWithNewReduction(reduction, num_tokens_per_input);

  std::string new_name = nextContextualRobeZOpName() + "_shared_" + name();
  return std::shared_ptr<ContextualRobeZ>(
      new ContextualRobeZ(std::move(new_kernel), new_name));
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
struct specialize<Archive, thirdai::bolt::nn::ops::ContextualRobeZ,
                  cereal::specialization::member_load_save> {};

}  // namespace cereal

CEREAL_REGISTER_TYPE(thirdai::bolt::nn::ops::ContextualRobeZ)