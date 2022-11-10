#pragma once

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
  DlrmAttentionNode() : _compiled_state(std::nullopt), _compiled(false) {}

  uint32_t outputDim() const final {
    if (getState() == NodeState::Constructed) {
      throw exceptions::NodeStateMachineError(
          "Cannot get the output dimension of DlrmAttentionNode before "
          "setting predecessors.");
    }

    return _compiled_state->_output_dim;
  }

  bool isInputNode() const final { return false; }

  std::shared_ptr<DlrmAttentionNode> setPredecessors(
      FullyConnectedNodePtr fully_connected_node,
      EmbeddingNodePtr embedding_node) {
    _fully_connected_node = std::move(fully_connected_node);
    _embedding_node = std::move(embedding_node);

    if ((_embedding_node->outputDim() % _fully_connected_node->outputDim()) !=
        0) {
      throw exceptions::GraphCompilationFailure(
          "Output dimension of EmbeddingNode must be a multiple of the output "
          "dimension of FullyConnectedNode in DLRMFeatureInteractionNode");
    }

    uint32_t num_embedding_chunks =
        _embedding_node->outputDim() / _fully_connected_node->outputDim();

    uint32_t output_dim = (num_embedding_chunks + 1) * num_embedding_chunks / 2;

    uint32_t embedding_chunk_size =
        _embedding_node->outputDim() / num_embedding_chunks;

    _compiled_state = CompiledState(
        /* num_embedding_chunks= */ num_embedding_chunks,
        /* output_dim= */ output_dim,
        /* embedding_chunk_size= */ embedding_chunk_size);

    return shared_from_this();
  }

 protected:
  void compileImpl() final { _compiled = true; }

  std::vector<std::shared_ptr<FullyConnectedLayer>>
  getInternalFullyConnectedLayersImpl() const final {
    return {};
  }

  void prepareForBatchProcessingImpl(uint32_t batch_size,
                                     bool use_sparsity) final {
    (void)use_sparsity;
    _outputs = BoltBatch(/* dim= */ _compiled_state->_output_dim,
                         /* batch_size= */ batch_size,
                         /* is_dense= */ true);
  }

  uint32_t numNonzerosInOutputImpl() const final { return outputDim(); }

  void forwardImpl(uint32_t vec_index, const BoltVector* labels) final {
    (void)labels;

    BoltVector& fc_output = _fully_connected_node->getOutputVector(vec_index);

    BoltVector& embedding_output = _embedding_node->getOutputVector(vec_index);

    BoltVector& output_vector = (*_outputs)[vec_index];

    // Compute interactions between outputs of fully connected layer and
    // embeddings.

    uint32_t embedding_chunk_size = _compiled_state->_embedding_chunk_size;

    for (uint32_t emb_idx = 0; emb_idx < _compiled_state->_num_embedding_chunks;
         emb_idx++) {
      if (fc_output.isDense()) {
        output_vector.activations[emb_idx] =
            fcOutputEmbeddingDotProduct</* FC_OUTPUT_DENSE= */ true>(
                fc_output,
                embedding_output.activations + emb_idx * embedding_chunk_size);
      } else {
        output_vector.activations[emb_idx] =
            fcOutputEmbeddingDotProduct</* FC_OUTPUT_DENSE= */ false>(
                fc_output,
                embedding_output.activations + emb_idx * embedding_chunk_size);
      }
    }

    // Compute pairwise interactions between embeddings.

    // TODO(Nicholas): is it faster to use eigen here since its more optimized
    // for dense computations, however it requires computing every pairwise dot
    // product twice?

    uint32_t output_idx = _compiled_state->_num_embedding_chunks;
    for (uint32_t emb_idx_1 = 0;
         emb_idx_1 < _compiled_state->_num_embedding_chunks; emb_idx_1++) {
      for (uint32_t emb_idx_2 = emb_idx_1 + 1;
           emb_idx_2 < _compiled_state->_num_embedding_chunks; emb_idx_2++) {
        output_vector.activations[output_idx++] = embeddingDotProduct(
            embedding_output.activations + emb_idx_1 * embedding_chunk_size,
            embedding_output.activations + emb_idx_2 * embedding_chunk_size,
            embedding_chunk_size);
      }
    }
  }

  void backpropagateImpl(uint32_t vec_index) final {
    BoltVector& fc_output = _fully_connected_node->getOutputVector(vec_index);

    BoltVector& embedding_output = _embedding_node->getOutputVector(vec_index);

    BoltVector& output_vector = (*_outputs)[vec_index];

    uint32_t embedding_chunk_size = _compiled_state->_embedding_chunk_size;

    for (uint32_t emb_idx = 0; emb_idx < _compiled_state->_num_embedding_chunks;
         emb_idx++) {
      float dot_product_gradient = output_vector.gradients[emb_idx];

      uint64_t embedding_offset = emb_idx * embedding_chunk_size;
      const float* embedding = embedding_output.activations + embedding_offset;
      float* embedding_grad = embedding_output.gradients + embedding_offset;

      if (fc_output.isDense()) {
        fcOutputEmbeddingDotProductBackward</* FC_OUTPUT_DENSE= */ true>(
            dot_product_gradient, fc_output, embedding, embedding_grad);
      } else {
        fcOutputEmbeddingDotProductBackward</* FC_OUTPUT_DENSE= */ false>(
            dot_product_gradient, fc_output, embedding, embedding_grad);
      }
    }

    uint32_t output_idx = _compiled_state->_num_embedding_chunks;
    for (uint32_t emb_idx_1 = 0;
         emb_idx_1 < _compiled_state->_num_embedding_chunks; emb_idx_1++) {
      for (uint32_t emb_idx_2 = emb_idx_1 + 1;
           emb_idx_2 < _compiled_state->_num_embedding_chunks; emb_idx_2++) {
        float dot_product_gradient = output_vector.gradients[output_idx++];

        uint64_t emb_1_offset = emb_idx_1 * embedding_chunk_size;
        const float* emb_1 = embedding_output.activations + emb_1_offset;
        float* emb_1_grad = embedding_output.gradients + emb_1_offset;

        uint64_t emb_2_offset = emb_idx_2 * embedding_chunk_size;
        const float* emb_2 = embedding_output.activations + emb_2_offset;
        float* emb_2_grad = embedding_output.gradients + emb_2_offset;

        embeddingDotProductBackward(dot_product_gradient, emb_1, emb_1_grad,
                                    emb_2, emb_2_grad, embedding_chunk_size);
      }
    }
  }

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

  void summarizeImpl(std::stringstream& summary, bool detailed) const final {
    (void)detailed;
    summary << "(" << _fully_connected_node->name() << ", "
            << _embedding_node->name() << ") -> " << name()
            << "(DLRMDotProductFeatureInteraction): output_dim="
            << _compiled_state->_output_dim << " num_embedding_chunks="
            << _compiled_state->_num_embedding_chunks << "\n";
  }

  // Return a short all lowercase string representing the type of this node for
  // use in printing the graph, e.g. concat, fc, input
  std::string type() const final { return "dlrm_feature_interaction"; }

  NodeState getState() const final {
    bool predecessors_set = _fully_connected_node && _embedding_node;
    if (!predecessors_set && !_compiled_state && !_compiled && !_outputs) {
      return NodeState::Constructed;
    }
    if (predecessors_set && _compiled_state && !_compiled && !_outputs) {
      return NodeState::PredecessorsSet;
    }
    if (predecessors_set && _compiled_state && _compiled && !_outputs) {
      return NodeState::Compiled;
    }
    if (predecessors_set && _compiled_state && _compiled && _outputs) {
      return NodeState::PreparedForBatchProcessing;
    }
    throw exceptions::NodeStateMachineError(
        "DLRMFeatureInteractionNode is in an invalid internal state");
  }

  bool trainable(bool flag) final {
    (void)flag;
    return false;
  }

 private:
  template <bool FC_OUTPUT_DENSE>
  static float fcOutputEmbeddingDotProduct(const BoltVector& fc_output,
                                           const float* const embedding) {
    float total = 0.0;
    for (uint32_t i = 0; i < fc_output.len; i++) {
      total += fc_output.activations[i] *
               embedding[fc_output.activeNeuronAtIndex<FC_OUTPUT_DENSE>(i)];
    }
    return total;
  }

  template <bool FC_OUTPUT_DENSE>
  static void fcOutputEmbeddingDotProductBackward(float dot_product_gradient,
                                                  const BoltVector& fc_output,
                                                  const float* const embedding,
                                                  float* const emb_gradient) {
    for (uint32_t i = 0; i < fc_output.len; i++) {
      uint32_t active_neuron =
          fc_output.activeNeuronAtIndex<FC_OUTPUT_DENSE>(i);
      fc_output.gradients[i] += dot_product_gradient * embedding[active_neuron];
      emb_gradient[active_neuron] +=
          dot_product_gradient * fc_output.activations[i];
    }
  }

  static float embeddingDotProduct(const float* const emb_1,
                                   const float* const emb_2, uint32_t dim) {
    float total = 0.0;
    for (uint32_t i = 0; i < dim; i++) {
      total += emb_1[i] * emb_2[i];
    }
    return total;
  }

  static void embeddingDotProductBackward(float dot_product_gradient,
                                          const float* const emb_1,
                                          float* const emb_1_grad,
                                          const float* const emb_2,
                                          float* const emb_2_grad,
                                          uint32_t dim) {
    for (uint32_t i = 0; i < dim; i++) {
      emb_1_grad[i] += dot_product_gradient * emb_2[i];
      emb_2_grad[i] += dot_product_gradient * emb_1[i];
    }
  }

  FullyConnectedNodePtr _fully_connected_node;
  EmbeddingNodePtr _embedding_node;

  struct CompiledState {
    explicit CompiledState(uint32_t num_embedding_chunks, uint32_t output_dim,
                           uint32_t embedding_chunk_size)
        : _num_embedding_chunks(num_embedding_chunks),
          _output_dim(output_dim),
          _embedding_chunk_size(embedding_chunk_size) {}

    uint32_t _num_embedding_chunks;
    uint32_t _output_dim;
    uint32_t _embedding_chunk_size;
  };
  std::optional<CompiledState> _compiled_state;

  bool _compiled;
  std::optional<BoltBatch> _outputs;
};

using DlrmAttentionNodePtr = std::shared_ptr<DlrmAttentionNode>;

}  // namespace thirdai::bolt
