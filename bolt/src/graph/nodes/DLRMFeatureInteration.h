#pragma once

#include <wrappers/src/EigenDenseWrapper.h>
#include "Embedding.h"
#include "FullyConnected.h"
#include <bolt/src/graph/Node.h>
#include <bolt/src/layers/BoltVector.h>
#include <Eigen/src/Core/Map.h>
#include <exceptions/src/Exceptions.h>
#include <optional>

namespace thirdai::bolt {

using EigenRowMajorMatrix =
    Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;

class DLRMFeatureInteractionNode final : public Node {
 public:
  DLRMFeatureInteractionNode() : _compiled_state(std::nullopt) {}

  uint32_t outputDim() const final {
    if (getState() != NodeState::Compiled) {
      throw exceptions::NodeStateMachineError(
          "Cannot get the output dimension of DLRMFeatureInteractionNode "
          "before the predecessors are set.");
    }

    return _compiled_state->_output_dim;
  }

  bool isInputNode() const final { return false; }

  void setPredecessors(FullyConnectedNodePtr fully_connected_node,
                       EmbeddingNodePtr embedding_node) {
    _fully_connected_node = std::move(fully_connected_node);
    _embedding_node = std::move(embedding_node);

    if ((_embedding_node->outputDim() % _fully_connected_node->outputDim()) !=
        0) {
      throw exceptions::GraphCompilationFailure(
          "Output dimension of EmbeddingNode must be a multiple of the output "
          "dimension of FullyConnectedNode in DLRMFeatureInteractionNode");
    }
  }

 protected:
  void compileImpl() final {
    uint32_t num_embedding_chunks =
        _embedding_node->outputDim() / _fully_connected_node->outputDim();

    /**
     * We want to compute the following pariwise dot products:
     *
     *            | fc_output | emb_1 | emb_2 | ... | emb_n
     *  fc_output |               X       X      X      X
     *      emb_1 |                       X      X      X
     *      emb_2 |                              X      X
     *       ...                                        X
     *      emb_n |
     *
     * We know that n = _num_embedding_chunks. Thus the number of pairwise
     * combinations of interest is (n+1)n/2.
     */
    uint32_t output_dim = (num_embedding_chunks + 1) * num_embedding_chunks / 2;

    _compiled_state = {num_embedding_chunks, output_dim};
  }

  std::vector<std::shared_ptr<FullyConnectedLayer>>
  getInternalFullyConnectedLayersImpl() const final {
    return {};
  }

  void prepareForBatchProcessingImpl(uint32_t batch_size,
                                     bool use_sparsity) final {
    (void)use_sparsity;
    _outputs = BoltBatch(batch_size, _compiled_state->_output_dim,
                         /* is_dense= */ false);
  }

  uint32_t numNonzerosInOutputImpl() const final { return outputDim(); }

  template <bool DENSE>
  static float dotProduct(const BoltVector& fc_output,
                          const float* const embedding) {
    float total = 0.0;
    for (uint32_t i = 0; i < fc_output.len; i++) {
      total += fc_output.activations[i] *
               embedding[fc_output.activeNeuronAtIndex<DENSE>(i)];
    }
    return total;
  }

  static float embeddingDotProduct(const float* const emb1,
                                   const float* const emb2, uint32_t dim) {
    float total = 0.0;
    for (uint32_t i = 0; i < dim; i++) {
      total += emb1[i] * emb2[i];
    }
    return total;
  }

  void forwardImpl(uint32_t vec_index, const BoltVector* labels) final {
    (void)labels;

    BoltVector& fc_output = _fully_connected_node->getOutputVector(vec_index);

    BoltVector& embedding_output = _embedding_node->getOutputVector(vec_index);

    BoltVector& output_vector = (*_outputs)[vec_index];

    // Compute interactions between outputs of fully connected layer and
    // embeddings.

    uint32_t embedding_chunk_size =
        embedding_output.len / _compiled_state->_num_embedding_chunks;

    for (uint32_t emb_idx = 0; emb_idx < _compiled_state->_num_embedding_chunks;
         emb_idx++) {
      if (fc_output.isDense()) {
        output_vector.activations[emb_idx] =
            dotProduct<true>(fc_output, embedding_output.activations +
                                            emb_idx * embedding_chunk_size);
      } else {
        output_vector.activations[emb_idx] =
            dotProduct<false>(fc_output, embedding_output.activations +
                                             emb_idx * embedding_chunk_size);
      }
    }

    // Compute pairwise interactions between embeddings.

    // TODO(Nicholas): is it faster to use eigen here since its more optimized
    // for dense computations, however it requires computing every pairwise dot
    // product twice?

    /** Eigen dot product computation
     * Eigen::Map<EigenRowMajorMatrix> eigen_embedding_chunks(
     *   embedding_output.activations, {_num_chunks, embedding_chunk_size});
     *
     * auto pairwise_dot_products =
     *    eigen_embedding_chunks * eigen_embedding_chunks.transpose();
     */

    uint32_t output_idx = _compiled_state->_num_embedding_chunks;
    for (uint32_t emb_idx_a = 0;
         emb_idx_a < _compiled_state->_num_embedding_chunks; emb_idx_a++) {
      for (uint32_t emb_idx_b = emb_idx_a + 1;
           emb_idx_b < _compiled_state->_num_embedding_chunks; emb_idx_b++) {
        output_vector.activations[output_idx++] = embeddingDotProduct(
            embedding_output.activations + emb_idx_a * embedding_chunk_size,
            embedding_output.activations + emb_idx_b * embedding_chunk_size,
            embedding_chunk_size);
      }
    }
  }

  void backpropagateImpl(uint32_t vec_index) final { (void)vec_index; }

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
            << _compiled_state->_num_embedding_chunks;
  }

  // Return a short all lowercase string representing the type of this node for
  // use in printing the graph, e.g. concat, fc, input
  std::string type() const final { return "dlrm_feature_interaction"; }

  NodeState getState() const final {
    bool predecessors_set = _fully_connected_node && _embedding_node;
    if (!predecessors_set && !_compiled_state && !_outputs) {
      return NodeState::Constructed;
    }
    if (predecessors_set && !_compiled_state && !_outputs) {
      return NodeState::PredecessorsSet;
    }
    if (predecessors_set && _compiled_state && !_outputs) {
      return NodeState::Compiled;
    }
    if (predecessors_set && _compiled_state && _outputs) {
      return NodeState::PreparedForBatchProcessing;
    }
    throw exceptions::NodeStateMachineError(
        "DLRMFeatureInteractionNode is in an invalid internal state");
  }

 private:
  FullyConnectedNodePtr _fully_connected_node;
  EmbeddingNodePtr _embedding_node;

  struct CompiledState {
    uint32_t _num_embedding_chunks;
    uint32_t _output_dim;
  };
  std::optional<CompiledState> _compiled_state;

  std::optional<BoltBatch> _outputs;
};

}  // namespace thirdai::bolt