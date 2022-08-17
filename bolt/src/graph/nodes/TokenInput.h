#pragma once

#include <cereal/access.hpp>
#include <bolt/src/graph/Node.h>
#include <dataset/src/batch_types/BoltTokenBatch.h>
#include <stdexcept>
#include <vector>

namespace thirdai::bolt {

/**
 * This class does not implement the Node interface because it is not
 * replaceable for other types of nodes like the regular Input node is. If a
 * node requires token input then it must take in a TokenInput node directly,
 * since no other node type will output tokens.
 */
class TokenInput : public Node {
 public:
  TokenInput() : _tokens(nullptr), _compiled(false) {}

  void setTokenInputs(dataset::BoltTokenBatch* tokens) { _tokens = tokens; }

  const std::vector<uint32_t>& getTokens(uint32_t vec_index) {
    return (*_tokens)[vec_index];
  }

  uint32_t outputDim() const final {
    throw std::logic_error("Cannot call outputDim() on TokenInput node");
  }

  bool isInputNode() const final { return true; }

  void prepareForTraining() {
    throw std::logic_error(
        "Should not call prepareForTraining() on TokenInput node");
  }

 private:
  void compileImpl() final { _compiled = true; }

  std::vector<std::shared_ptr<FullyConnectedLayer>>
  getInternalFullyConnectedLayersImpl() const final {
    return {};
  }

  void prepareForBatchProcessingImpl(uint32_t batch_size,
                                     bool use_sparsity) final {
    (void)batch_size;
    (void)use_sparsity;
    throw exceptions::NodeStateMachineError(
        "Should never call prepareForBatchProcessing on TokenInput (instead "
        "should call setInputs).");
  }

  uint32_t numNonzerosInOutputImpl() const final {
    throw std::logic_error(
        "Cannot call numNonzerosInOutput() on TokenInput node.");
  }

  void forwardImpl(uint32_t vec_index, const BoltVector* labels) final {
    (void)labels;
    (void)vec_index;
  }

  void backpropagateImpl(uint32_t vec_index) final { (void)vec_index; }

  void updateParametersImpl(float learning_rate, uint32_t batch_cnt) final {
    (void)learning_rate;
    (void)batch_cnt;
  }

  BoltVector& getOutputVectorImpl(uint32_t vec_index) final {
    (void)vec_index;
    throw std::logic_error("Cannot call getOutputVector() on TokenInput node.");
  }

  void cleanupAfterBatchProcessingImpl() final { _tokens = nullptr; }

  void summarizeImpl(std::stringstream& summary, bool detailed) const final {
    (void)detailed;
    summary << name() << " (TokenInput)\n";
  }

  std::string type() const final { return "token_input"; }

  std::vector<NodePtr> getPredecessorsImpl() const final { return {}; }

  NodeState getState() const final {
    if (!_compiled && _tokens == nullptr) {
      return NodeState::PredecessorsSet;
    }
    if (_compiled && _tokens == nullptr) {
      return NodeState::Compiled;
    }
    if (_compiled && _tokens != nullptr) {
      return NodeState::PreparedForBatchProcessing;
    }
    throw exceptions::NodeStateMachineError(
        "TokenInputNode is in an invalid internal state");
  }

  friend class cereal::access;
  template <class Archive>
  void serialize(Archive& archive) {
    archive(cereal::base_class<Node>(this), _compiled);
  }

  dataset::BoltTokenBatch* _tokens;
  bool _compiled;
};

using TokenInputPtr = std::shared_ptr<TokenInput>;

}  // namespace thirdai::bolt

CEREAL_REGISTER_TYPE(thirdai::bolt::TokenInput)
