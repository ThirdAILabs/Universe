#pragma once

#include <bolt/src/nn/ops/Op.h>
#include <memory>

namespace thirdai::bolt {

/**
 * This op computes the pairwise dot products between the output of a fully
 * connected layer and chunks of an embedding. The embedding is broken into
 * chunks of dimension the same as the fuly connected output. Then the following
 * dot products are computed.
 *
 *            | fc_output | emb_1 | emb_2 | ... | emb_n
 *  fc_output |               X       X      X      X
 *      emb_1 |                       X      X      X
 *      emb_2 |                              X      X
 *       ...                                        X
 *      emb_n |
 *
 * If the embedding layer uses concatenation then ideally the fully connected
 * layer will have a dimension that is the same as the dimension of the
 * embedding for each token.
 *
 * Thus the number of pairwise combinations of interest is (n+1)n/2, meaning the
 * output of the layer is (n+1)n/2.
 */
class DlrmAttention final : public Op,
                            public std::enable_shared_from_this<DlrmAttention> {
 public:
  static std::shared_ptr<DlrmAttention> make();

  void forward(const ComputationList& inputs, TensorPtr& output,
               uint32_t index_in_batch, bool training) final;

  void backpropagate(ComputationList& inputs, TensorPtr& output,
                     uint32_t index_in_batch) final;

  void updateParameters(float learning_rate, uint32_t train_steps) final {
    (void)learning_rate;
    (void)train_steps;
  }

  uint32_t dim() const final;

  std::optional<uint32_t> nonzeros(const ComputationList& inputs,
                                   bool use_sparsity) const final;

  void initOptimizer() final;

  void disableSparseParameterUpdates() final {}

  void enableSparseParameterUpdates() final {}

  std::vector<std::vector<float>*> gradients() final { return {}; }

  std::vector<std::vector<float>*> parameters() final { return {}; }

  ComputationPtr applyToInputs(const ComputationList& inputs) final;

  ar::ConstArchivePtr toArchive(bool with_optimizer) const final;

  void summary(std::ostream& summary, const ComputationList& inputs,
               const Computation* output) const final;

  ComputationPtr apply(ComputationPtr fc_input, ComputationPtr emb_input);

 private:
  DlrmAttention() {}

  template <bool FC_OUTPUT_DENSE>
  static float fcOutputEmbeddingDotProduct(const BoltVector& fc_output,
                                           const float* embedding);
  template <bool FC_OUTPUT_DENSE>
  static void fcOutputEmbeddingDotProductBackward(float dot_product_gradient,
                                                  const BoltVector& fc_output,
                                                  const float* embedding,
                                                  float* emb_gradient);

  static float embeddingDotProduct(const float* emb_1, const float* emb_2,
                                   uint32_t dim);

  static void embeddingDotProductBackward(float dot_product_gradient,
                                          const float* emb_1, float* emb_1_grad,
                                          const float* emb_2, float* emb_2_grad,
                                          uint32_t dim);

  uint32_t _n_emb_chunks = 0;
  uint32_t _emb_chunk_size = 0;
};

using DlrmAttentionPtr = std::shared_ptr<DlrmAttention>;

}  // namespace thirdai::bolt