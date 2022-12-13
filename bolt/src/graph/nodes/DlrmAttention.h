#pragma once

#include <cereal/access.hpp>
#include <cereal/types/base_class.hpp>
#include <cereal/types/memory.hpp>
#include <cereal/types/optional.hpp>
#include <cereal/types/polymorphic.hpp>
#include "Embedding.h"
#include "FullyConnected.h"
#include <bolt/src/graph/Node.h>
#include <exceptions/src/Exceptions.h>
#include <memory>
#include <optional>

namespace thirdai::bolt {

/**
 * This layer computes the pairwise dot products between the output of a fully
 * connected layers and an embedding. The embedding is broken into chunks of
 * dimension the same as the fuly connected output. Then the following dot
 * products are computed.
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
 * The the number of pairwise combinations of interest is (n+1)n/2, meaning the
 * output of the layer is (n+1)n/2.
 */
class DlrmAttentionNode final
    : public Node,
      public std::enable_shared_from_this<DlrmAttentionNode> {
 public:
  DlrmAttentionNode()
      : _compiled_state(std::nullopt),
        _compiled(false),
        _outputs(std::nullopt) {}

  uint32_t outputDim() const final;

  bool isInputNode() const final { return false; }

  std::shared_ptr<DlrmAttentionNode> setPredecessors(
      FullyConnectedNodePtr fully_connected_node,
      EmbeddingNodePtr embedding_node);

 protected:
  void compileImpl() final { _compiled = true; }

  std::vector<std::shared_ptr<FullyConnectedLayer>>
  getInternalFullyConnectedLayersImpl() const final {
    return {};
  }

  void prepareForBatchProcessingImpl(uint32_t batch_size,
                                     bool use_sparsity) final;

  uint32_t numNonzerosInOutputImpl() const final { return outputDim(); }

  void forwardImpl(uint32_t vec_index, const BoltVector* labels) final;

  void backpropagateImpl(uint32_t vec_index) final;

  void initOptimizer() final {}

  void updateParametersImpl(float learning_rate, uint32_t batch_cnt) final {
    (void)learning_rate;
    (void)batch_cnt;
  }

  BoltVector& getOutputVectorImpl(uint32_t vec_index) final {
    return (*_outputs)[vec_index];
  }

  void cleanupAfterBatchProcessingImpl() final { _outputs = std::nullopt; }

  std::vector<NodePtr> getPredecessorsImpl() const final {
    return {_fully_connected_node, _embedding_node};
  }

  void summarizeImpl(std::stringstream& summary, bool detailed) const final;

  // Return a short all lowercase string representing the type of this node for
  // use in printing the graph, e.g. concat, fc, input
  std::string type() const final { return "dlrm_feature_interaction"; }

  NodeState getState() const final;

  bool needGradientSharing() final { return false; }

 private:
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

  friend class cereal::access;
  template <class Archive>
  void serialize(Archive& archive) {
    archive(cereal::base_class<Node>(this), _fully_connected_node,
            _embedding_node, _compiled_state, _compiled);
  }

  FullyConnectedNodePtr _fully_connected_node;
  EmbeddingNodePtr _embedding_node;

  struct CompiledState {
    explicit CompiledState(uint32_t num_embedding_chunks, uint32_t output_dim,
                           uint32_t embedding_chunk_size)
        : _num_embedding_chunks(num_embedding_chunks),
          _output_dim(output_dim),
          _embedding_chunk_size(embedding_chunk_size) {}

    // This needs to be a public constructor so that when cereal constructs an
    // optional of CompiledState and calls emplace,
    // std::allocator<CompiledState> can construct the object. See e.g.
    // https://stackoverflow.com/questions/17007977/vectoremplace-back-for-objects-with-a-private-constructor
    // You shouldn't call this function yourself.
    CompiledState()
        : _num_embedding_chunks(0), _output_dim(0), _embedding_chunk_size(0) {}

    uint32_t _num_embedding_chunks;
    uint32_t _output_dim;
    uint32_t _embedding_chunk_size;

   private:
    friend class cereal::access;
    template <class Archive>
    void serialize(Archive& archive) {
      archive(_num_embedding_chunks, _output_dim, _embedding_chunk_size);
    }
  };
  std::optional<CompiledState> _compiled_state;

  bool _compiled;
  std::optional<BoltBatch> _outputs;
};

using DlrmAttentionNodePtr = std::shared_ptr<DlrmAttentionNode>;

}  // namespace thirdai::bolt
