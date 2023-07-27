#include "MultiplicativePositionalEmbedding.h"
#include <cereal/archives/binary.hpp>
#include <cereal/specialize.hpp>
#include <cereal/types/base_class.hpp>
#include <cereal/types/memory.hpp>
#include <cereal/types/polymorphic.hpp>
#include <bolt/src/layers/EmbeddingLayer.h>
#include <bolt/src/layers/LayerConfig.h>
#include <bolt/src/nn/autograd/Computation.h>
#include <bolt/src/nn/ops/Op.h>
#include <bolt_vector/src/BoltVector.h>
#include <memory>

namespace thirdai::bolt::nn::ops {
std::string nextMultiplicativePositionalEncodingOpName() {
  static uint32_t constructed = 0;
  return "multiplicative_posemb_" + std::to_string(++constructed);
}

MultiplicativePosEmbedding::MultiplicativePosEmbedding(
    uint64_t num_embedding_lookups, uint64_t lookup_size,
    uint64_t log_embedding_block_size, const std::string& reduction,
    std::optional<uint64_t> num_tokens_per_input, uint64_t update_chunk_size)
    : Op(nextMultiplicativePositionalEncodingOpName()) {
  EmbeddingLayerConfig token_config(
      /* num_embedding_lookups= */ num_embedding_lookups,
      /* lookup_size= */ lookup_size,
      /* log_embedding_block_size= */ log_embedding_block_size,
      /* update_chunk_size= */ update_chunk_size, /* reduction= */ reduction,
      /* num_tokens_per_input= */ num_tokens_per_input);
  _token_kernel = std::make_unique<EmbeddingLayer>(token_config);
  EmbeddingLayerConfig pos_config(
      /* num_embedding_lookups= */ num_embedding_lookups,
      /* lookup_size= */ lookup_size,
      /* log_embedding_block_size= */ log_embedding_block_size,
      /* update_chunk_size= */ update_chunk_size, /* reduction= */ reduction,
      /* num_tokens_per_input= */ num_tokens_per_input);
  _pos_kernel = std::make_unique<EmbeddingLayer>(pos_config);
}

std::shared_ptr<MultiplicativePosEmbedding> MultiplicativePosEmbedding::make(
    uint64_t num_embedding_lookups, uint64_t lookup_size,
    uint64_t log_embedding_block_size, const std::string& reduction,
    std::optional<uint64_t> num_tokens_per_input, uint64_t update_chunk_size) {
  return std::shared_ptr<MultiplicativePosEmbedding>(
      new MultiplicativePosEmbedding(
          /* num_embedding_lookups= */ num_embedding_lookups,
          /* lookup_size= */ lookup_size,
          /* log_embedding_block_size= */ log_embedding_block_size,
          /* reduction= */ reduction,
          /* num_tokens_per_input= */ num_tokens_per_input,
          /* update_chunk_size= */ update_chunk_size));
}

void MultiplicativePosEmbedding::forward(
    const autograd::ComputationList& inputs, tensor::TensorPtr& output,
    uint32_t index_in_batch, bool training) {
  (void)training;
  assert(inputs.size() == 1);
  BoltVector position_output = output->getVector(index_in_batch).copy();
  BoltVector position_input =
      inputs[0]->tensor()->getVector(index_in_batch).copy();
  for (uint64_t token_idx = 0; token_idx < position_input.len; token_idx++) {
    position_input.active_neurons[token_idx] = token_idx;
  }
  _pos_kernel->forward(position_input, position_output);
  _token_kernel->forward(inputs[0]->tensor()->getVector(index_in_batch),
                         output->getVector(index_in_batch));
  BoltVector& output_boltvector = output->getVector(index_in_batch);

  assert(output_boltvector.len == position_output.len);
  for (uint64_t token_index = 0; token_index < output_boltvector.len;
       token_index++) {
    output_boltvector.activations[token_index] *=
        position_output.activations[token_index];
  }
}

void MultiplicativePosEmbedding::backpropagate(
    autograd::ComputationList& inputs, tensor::TensorPtr& output,
    uint32_t index_in_batch) {
  assert(inputs.size() == 1);

  auto output_token = output->getVector(index_in_batch).copy();
  auto output_position = output->getVector(index_in_batch).copy();

  for (uint32_t index = 0; index < output_token.len; index++) {
    output_token.gradients[index] *= output_position.activations[index];
    output_position.gradients[index] *= output_token.activations[index];
  }

  _token_kernel->backpropagate(inputs[0]->tensor()->getVector(index_in_batch),
                               output_token);
  BoltVector position_input =
      inputs[0]->tensor()->getVector(index_in_batch).copy();
  for (uint64_t token_idx = 0; token_idx < position_input.len; token_idx++) {
    position_input.active_neurons[token_idx] = token_idx;
  }
  _pos_kernel->backpropagate(position_input, output_position);
}

void MultiplicativePosEmbedding::updateParameters(float learning_rate,
                                                  uint32_t train_steps) {
  _pos_kernel->updateParameters(learning_rate, train_steps, BETA1, BETA2, EPS);
  _token_kernel->updateParameters(learning_rate, train_steps, BETA1, BETA2,
                                  EPS);
}

uint32_t MultiplicativePosEmbedding::dim() const {
  assert(_pos_kernel->getOutputDim() == _token_kernel->getOutputDim());
  return _pos_kernel->getOutputDim();
}

std::optional<uint32_t> MultiplicativePosEmbedding::nonzeros(
    const autograd::ComputationList& inputs, bool use_sparsity) const {
  (void)inputs;
  (void)use_sparsity;

  return dim();
}

void MultiplicativePosEmbedding::disableSparseParameterUpdates() {
  _pos_kernel->disableSparseParameterUpdates();
  _token_kernel->disableSparseParameterUpdates();
}

void MultiplicativePosEmbedding::enableSparseParameterUpdates() {
  _pos_kernel->enableSparseParameterUpdates();
  _token_kernel->enableSparseParameterUpdates();
}

std::vector<std::vector<float>*> MultiplicativePosEmbedding::gradients() {
  return {&_pos_kernel->getRawEmbeddingBlockGradient(),
          &_token_kernel->getRawEmbeddingBlockGradient()};
}

std::vector<std::vector<float>*> MultiplicativePosEmbedding::parameters() {
  return {&_pos_kernel->getRawEmbeddingBlock(),
          &_token_kernel->getRawEmbeddingBlock()};
}

void MultiplicativePosEmbedding::summary(
    std::ostream& summary, const autograd::ComputationList& inputs,
    const autograd::Computation* output) const {
  summary << "MultiplicativePosEmbedding(" << name()
          << "): " << inputs[0]->name() << " -> " << output->name()
          << " [pos_kernel : ";
  _pos_kernel->buildLayerSummary(summary);
  summary << "token_kernel : ";
  _token_kernel->buildLayerSummary(summary);
  summary << "]";
}

void MultiplicativePosEmbedding::setSerializeOptimizer(
    bool should_serialize_optimizer) {
  _pos_kernel->saveWithOptimizer(should_serialize_optimizer);
  _token_kernel->saveWithOptimizer(should_serialize_optimizer);
}

autograd::ComputationPtr MultiplicativePosEmbedding::apply(
    autograd::ComputationPtr input) {
  return autograd::Computation::make(shared_from_this(), {std::move(input)});
}

template void MultiplicativePosEmbedding::save(
    cereal::BinaryOutputArchive&) const;

template <class Archive>
void MultiplicativePosEmbedding::save(Archive& archive) const {
  archive(cereal::base_class<Op>(this), _pos_kernel, _token_kernel);
}

template void MultiplicativePosEmbedding::load(cereal::BinaryInputArchive&);

template <class Archive>
void MultiplicativePosEmbedding::load(Archive& archive) {
  archive(cereal::base_class<Op>(this), _pos_kernel, _token_kernel);
  _pos_kernel->initOptimizer();
  _token_kernel->initOptimizer();
}

std::shared_ptr<MultiplicativePosEmbedding>
MultiplicativePosEmbedding::duplicateWithNewReduction(
    const std::string& reduction,
    std::optional<uint64_t> num_tokens_per_input) {
  auto new_pos_kernel =
      _pos_kernel->duplicateWithNewReduction(reduction, num_tokens_per_input);
  auto new_token_kernel =
      _token_kernel->duplicateWithNewReduction(reduction, num_tokens_per_input);
  std::string new_name =
      nextMultiplicativePositionalEncodingOpName() + "_shared_" + name();
  return std::shared_ptr<MultiplicativePosEmbedding>(
      new MultiplicativePosEmbedding(std::move(new_pos_kernel),
                                     std::move(new_token_kernel), new_name));
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
struct specialize<Archive, thirdai::bolt::nn::ops::MultiplicativePosEmbedding,
                  cereal::specialization::member_load_save> {};

}  // namespace cereal

CEREAL_REGISTER_TYPE(thirdai::bolt::nn::ops::MultiplicativePosEmbedding)