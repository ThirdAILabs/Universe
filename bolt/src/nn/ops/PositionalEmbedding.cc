#include "PositionalEmbedding.h"
#include <cereal/archives/binary.hpp>
#include <cereal/specialize.hpp>
#include <cereal/types/base_class.hpp>
#include <cereal/types/memory.hpp>
#include <bolt/src/layers/EmbeddingLayer.h>
#include <bolt/src/layers/LayerConfig.h>
#include <bolt/src/nn/autograd/Computation.h>
#include <bolt/src/nn/ops/Op.h>
#include <bolt_vector/src/BoltVector.h>
#include <memory>

namespace thirdai::bolt::nn::ops {
std::string nextPositionalEncodingOpName() {
  static uint32_t constructed = 0;
  return "posemb_" + std::to_string(++constructed);
}

PosEmbedding::PosEmbedding(uint64_t num_embedding_lookups, uint64_t lookup_size,
                           uint64_t log_embedding_block_size,
                           const std::string& reduction,
                           std::optional<uint64_t> num_tokens_per_input,
                           uint64_t update_chunk_size)
    : Op(nextPositionalEncodingOpName()) {
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
  _pos_kernel = std::make_unique<EmbeddingLayer>(token_config);
}

std::shared_ptr<PosEmbedding> PosEmbedding::make(
    uint64_t num_embedding_lookups, uint64_t lookup_size,
    uint64_t log_embedding_block_size, const std::string& reduction,
    std::optional<uint64_t> num_tokens_per_input, uint64_t update_chunk_size) {
  return std::shared_ptr<PosEmbedding>(new PosEmbedding(
      /* num_embedding_lookups= */ num_embedding_lookups,
      /* lookup_size= */ lookup_size,
      /* log_embedding_block_size= */ log_embedding_block_size,
      /* reduction= */ reduction,
      /* num_tokens_per_input= */ num_tokens_per_input,
      /* update_chunk_size= */ update_chunk_size));
}

void PosEmbedding::forward(const autograd::ComputationList& inputs,
                           tensor::TensorPtr& output, uint32_t index_in_batch,
                           bool training) {
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
    output_boltvector.activations[token_index] +=
        position_output.activations[token_index];
  }
}

void PosEmbedding::backpropagate(autograd::ComputationList& inputs,
                                 tensor::TensorPtr& output,
                                 uint32_t index_in_batch) {
  assert(inputs.size() == 1);
  _token_kernel->backpropagate(inputs[0]->tensor()->getVector(index_in_batch),
                               output->getVector(index_in_batch));
  BoltVector position_input =
      inputs[0]->tensor()->getVector(index_in_batch).copy();
  for (uint64_t token_idx = 0; token_idx < position_input.len; token_idx++) {
    position_input.active_neurons[token_idx] = token_idx;
  }
  _pos_kernel->backpropagate(position_input, output->getVector(index_in_batch));
}

void PosEmbedding::updateParameters(float learning_rate, uint32_t train_steps) {
  _pos_kernel->updateParameters(learning_rate, train_steps, BETA1, BETA2, EPS);
  _token_kernel->updateParameters(learning_rate, train_steps, BETA1, BETA2,
                                  EPS);
}

uint32_t PosEmbedding::dim() const {
  assert(_pos_kernel->getOutputDim() == _token_kernel->getOutputDim());
  return _pos_kernel->getOutputDim();
}

std::optional<uint32_t> PosEmbedding::nonzeros(
    const autograd::ComputationList& inputs, bool use_sparsity) const {
  (void)inputs;
  (void)use_sparsity;

  return dim();
}

void PosEmbedding::disableSparseParameterUpdates() {
  _pos_kernel->disableSparseParameterUpdates();
  _token_kernel->disableSparseParameterUpdates();
}

std::vector<std::vector<float>*> PosEmbedding::gradients() {
  return {&_pos_kernel->getRawEmbeddingBlockGradient(),
          &_token_kernel->getRawEmbeddingBlockGradient()};
}

std::vector<std::vector<float>*> PosEmbedding::parameters() {
  return {&_pos_kernel->getRawEmbeddingBlock(),
          &_token_kernel->getRawEmbeddingBlock()};
}
}  // namespace thirdai::bolt::nn::ops