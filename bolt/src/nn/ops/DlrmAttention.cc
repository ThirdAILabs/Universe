#include "DlrmAttention.h"
#include "bolt/src/nn/autograd/Computation.h"
#include <bolt/src/nn/ops/Op.h>
#include <bolt_vector/src/BoltVector.h>
#include <memory>
#include <stdexcept>

namespace thirdai::bolt {

std::shared_ptr<DlrmAttention> DlrmAttention::make() {
  return std::shared_ptr<DlrmAttention>(new DlrmAttention());
}

void DlrmAttention::forward(const ComputationList& inputs, TensorPtr& output,
                            uint32_t index_in_batch, bool training) {
  assert(inputs.size() == 2);
  (void)training;

  const BoltVector& fc_input =
      inputs.at(0)->tensor()->getVector(index_in_batch);
  const BoltVector& emb_input =
      inputs.at(1)->tensor()->getVector(index_in_batch);

  BoltVector& output_vec = output->getVector(index_in_batch);

  // Compute interactions between outputs of fully connected layer and
  // embeddings.
  uint32_t output_idx = 0;
  for (uint32_t emb_idx = 0; emb_idx < _n_emb_chunks; emb_idx++) {
    if (fc_input.isDense()) {
      output_vec.activations[output_idx++] =
          fcOutputEmbeddingDotProduct</* FC_OUTPUT_DENSE= */ true>(
              fc_input, emb_input.activations + emb_idx * _emb_chunk_size);
    } else {
      output_vec.activations[output_idx++] =
          fcOutputEmbeddingDotProduct</* FC_OUTPUT_DENSE= */ false>(
              fc_input, emb_input.activations + emb_idx * _emb_chunk_size);
    }
  }

  // Compute pairwise interactions between embeddings.
  for (uint32_t emb_idx_1 = 0; emb_idx_1 < _n_emb_chunks; emb_idx_1++) {
    for (uint32_t emb_idx_2 = emb_idx_1 + 1; emb_idx_2 < _n_emb_chunks;
         emb_idx_2++) {
      output_vec.activations[output_idx++] = embeddingDotProduct(
          emb_input.activations + emb_idx_1 * _emb_chunk_size,
          emb_input.activations + emb_idx_2 * _emb_chunk_size, _emb_chunk_size);
    }
  }
}

void DlrmAttention::backpropagate(ComputationList& inputs, TensorPtr& output,
                                  uint32_t index_in_batch) {
  BoltVector& fc_input = inputs.at(0)->tensor()->getVector(index_in_batch);
  BoltVector& emb_input = inputs.at(1)->tensor()->getVector(index_in_batch);

  const BoltVector& output_vec = output->getVector(index_in_batch);

  uint32_t output_idx = 0;
  for (uint32_t emb_idx = 0; emb_idx < _n_emb_chunks; emb_idx++) {
    float dot_product_gradient = output_vec.gradients[output_idx++];

    uint64_t embedding_offset = emb_idx * _emb_chunk_size;
    const float* embedding = emb_input.activations + embedding_offset;
    float* embedding_grad = emb_input.gradients + embedding_offset;

    if (fc_input.isDense()) {
      fcOutputEmbeddingDotProductBackward</* FC_OUTPUT_DENSE= */ true>(
          dot_product_gradient, fc_input, embedding, embedding_grad);
    } else {
      fcOutputEmbeddingDotProductBackward</* FC_OUTPUT_DENSE= */ false>(
          dot_product_gradient, fc_input, embedding, embedding_grad);
    }
  }

  for (uint32_t emb_idx_1 = 0; emb_idx_1 < _n_emb_chunks; emb_idx_1++) {
    for (uint32_t emb_idx_2 = emb_idx_1 + 1; emb_idx_2 < _n_emb_chunks;
         emb_idx_2++) {
      float dot_product_gradient = output_vec.gradients[output_idx++];

      uint64_t emb_1_offset = emb_idx_1 * _emb_chunk_size;
      const float* emb_1 = emb_input.activations + emb_1_offset;
      float* emb_1_grad = emb_input.gradients + emb_1_offset;

      uint64_t emb_2_offset = emb_idx_2 * _emb_chunk_size;
      const float* emb_2 = emb_input.activations + emb_2_offset;
      float* emb_2_grad = emb_input.gradients + emb_2_offset;

      embeddingDotProductBackward(dot_product_gradient, emb_1, emb_1_grad,
                                  emb_2, emb_2_grad, _emb_chunk_size);
    }
  }
}

uint32_t DlrmAttention::dim() const {
  return (_n_emb_chunks + 1) * _n_emb_chunks / 2;
}

std::optional<uint32_t> DlrmAttention::nonzeros(const ComputationList& inputs,
                                                bool use_sparsity) const {
  (void)inputs;
  (void)use_sparsity;

  return dim();
}

void DlrmAttention::initOptimizer() {}

void DlrmAttention::summary(std::ostream& summary,
                            const ComputationList& inputs,
                            const Computation* output) const {
  summary << "DlrmAttention(" << name() << "): " << inputs.at(0)->name() << ", "
          << inputs.at(1)->name() << ") -> " << output->name()
          << "[n_emb_chunks=" << _n_emb_chunks
          << ", emb_chunk_size=" << _emb_chunk_size << "]";
}

template <bool FC_OUTPUT_DENSE>
float DlrmAttention::fcOutputEmbeddingDotProduct(const BoltVector& fc_output,
                                                 const float* embedding) {
  float total = 0.0;
  for (uint32_t i = 0; i < fc_output.len; i++) {
    total += fc_output.activations[i] *
             embedding[fc_output.activeNeuronAtIndex<FC_OUTPUT_DENSE>(i)];
  }
  return total;
}

template <bool FC_OUTPUT_DENSE>
void DlrmAttention::fcOutputEmbeddingDotProductBackward(
    float dot_product_gradient, const BoltVector& fc_output,
    const float* embedding, float* emb_gradient) {
  for (uint32_t i = 0; i < fc_output.len; i++) {
    uint32_t active_neuron = fc_output.activeNeuronAtIndex<FC_OUTPUT_DENSE>(i);
    fc_output.gradients[i] += dot_product_gradient * embedding[active_neuron];
    emb_gradient[active_neuron] +=
        dot_product_gradient * fc_output.activations[i];
  }
}

float DlrmAttention::embeddingDotProduct(const float* emb_1, const float* emb_2,
                                         uint32_t dim) {
  float total = 0.0;
  for (uint32_t i = 0; i < dim; i++) {
    total += emb_1[i] * emb_2[i];
  }
  return total;
}

void DlrmAttention::embeddingDotProductBackward(
    float dot_product_gradient, const float* emb_1, float* emb_1_grad,
    const float* emb_2, float* emb_2_grad, uint32_t dim) {
  for (uint32_t i = 0; i < dim; i++) {
    emb_1_grad[i] += dot_product_gradient * emb_2[i];
    emb_2_grad[i] += dot_product_gradient * emb_1[i];
  }
}

ComputationPtr DlrmAttention::apply(ComputationPtr fc_input,
                                    ComputationPtr emb_input) {
  uint32_t fc_dim = fc_input->dim();
  uint32_t emb_dim = emb_input->dim();

  if ((emb_dim % fc_dim) != 0) {
    throw std::invalid_argument(
        "Embedding dimension must be a multiple of the fully connected label "
        "dimension.");
  }

  if (emb_input->dim() != emb_input->nonzeros(/* use_sparsity= */ true)) {
    throw std::invalid_argument("Expected embedding to be dense.");
  }

  uint32_t n_emb_chunks = emb_dim / fc_dim;
  uint32_t emb_chunk_size = emb_dim / n_emb_chunks;

  if (_n_emb_chunks == 0) {
    _n_emb_chunks = n_emb_chunks;
    _emb_chunk_size = emb_chunk_size;
  } else if (n_emb_chunks != _emb_chunk_size ||
             emb_chunk_size != _emb_chunk_size) {
    std::stringstream error;

    error << "Expected DlrmAttention op to be applied to inputs "
             "yielding "
          << _n_emb_chunks << " chunks, each with dimension " << _emb_chunk_size
          << " but received inputs yielding " << n_emb_chunks
          << " chunks with dimension " << emb_chunk_size << ".";

    throw std::invalid_argument(error.str());
  }

  return Computation::make(shared_from_this(),
                           {std::move(fc_input), std::move(emb_input)});
}

}  // namespace thirdai::bolt