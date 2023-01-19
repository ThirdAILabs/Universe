#pragma once

#include "FullyConnected.h"
#include "Input.h"
#include <bolt/src/graph/Node.h>
#include <exceptions/src/Exceptions.h>
#include <memory>
#include <utility>
#include <vector>

namespace thirdai::bolt {

/**
 * This class is used in MLM experiments. This node stores N fully connected
 * layers, and takes in a regular bolt vector input as well as a token. For each
 * input there should be only a single token in the range [0,N) that indicates
 * which of the fully connected layers to use for the input. For instance in a
 * MLM model this was used where the number of layers was equal to the maximum
 * number of tokens in the sentence and for each input the layer at the index of
 * the masked token was used. The idea being that it would learn a slightly
 * different representation for each token in the sentence.
 */
class SwitchNode final : public Node,
                         public std::enable_shared_from_this<SwitchNode> {
 private:
  SwitchNode(uint32_t dim, const std::string& activation, uint32_t n_layers);

  SwitchNode(uint32_t dim, float sparsity, const std::string& activation,
             uint32_t n_layers);

  SwitchNode(uint32_t dim, float sparsity, const std::string& activation,
             const SamplingConfigPtr& sampling_config, uint32_t n_layers);

 public:
  static std::shared_ptr<SwitchNode> makeDense(uint32_t dim,
                                               const std::string& activation,
                                               uint32_t n_layers) {
    return std::shared_ptr<SwitchNode>(
        new SwitchNode(dim, activation, n_layers));
  }

  static std::shared_ptr<SwitchNode> makeAutotuned(
      uint32_t dim, float sparsity, const std::string& activation,
      uint32_t n_layers) {
    return std::shared_ptr<SwitchNode>(
        new SwitchNode(dim, sparsity, activation, n_layers));
  }

  static std::shared_ptr<SwitchNode> make(
      uint32_t dim, float sparsity, const std::string& activation,
      const SamplingConfigPtr& sampling_config, uint32_t n_layers) {
    return std::shared_ptr<SwitchNode>(
        new SwitchNode(dim, sparsity, activation, sampling_config, n_layers));
  }

  uint32_t outputDim() const final {
    // All layers are constructed identically so we can use _layers[0] here.
    return _layers.at(0)->outputDim();
  }

  bool isInputNode() const final { return false; }

  void initOptimizer() final;

  std::shared_ptr<SwitchNode> addPredecessors(NodePtr predecessor,  // NOLINT
                                              InputPtr token_input);

  void disableSparseParameterUpdates() final;

  void nodeSaveType(bool whether_hard_save) final;

  bool hasParameters() final { return false; }

 private:
  void compileImpl() final;

  std::vector<std::shared_ptr<FullyConnectedLayer>>
  getInternalFullyConnectedLayersImpl() const final;

  void prepareForBatchProcessingImpl(uint32_t batch_size,
                                     bool use_sparsity) final;

  uint32_t numNonzerosInOutputImpl() const final {
    // All layers are constructed identically so we can use _layers[0] here.
    return _layers.at(0)->numNonzerosInOutput();
  }

  void forwardImpl(uint32_t vec_index, const BoltVector* labels) final;

  void backpropagateImpl(uint32_t vec_index) final;

  void updateParametersImpl(float learning_rate, uint32_t batch_cnt) final;

  BoltVector& getOutputVectorImpl(uint32_t vec_index) final;

  void cleanupAfterBatchProcessingImpl() final;

  void summarizeImpl(std::stringstream& summary, bool detailed) const final;

  std::string type() const final { return "switch"; }

  std::vector<NodePtr> getPredecessorsImpl() const final;

  NodeState getState() const final;

  uint32_t getActiveLayer(uint32_t vec_index);

  friend class cereal::access;
  template <class Archive>
  void serialize(Archive& archive) {
    archive(cereal::base_class<Node>(this), _layers, _layers_used,
            _token_input);
  }

  std::vector<std::shared_ptr<FullyConnectedNode>> _layers;
  std::vector<bool> _layers_used;
  InputPtr _token_input;
};

}  // namespace thirdai::bolt