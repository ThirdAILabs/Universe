#pragma once

#include <cereal/access.hpp>
#include <bolt/src/graph/Node.h>
#include <bolt_vector/src/BoltVector.h>
#include <memory>
#include <stdexcept>

namespace thirdai::bolt {

class SparsificationNode final
    : public Node,
      public std::enable_shared_from_this<SparsificationNode> {
 private:
  explicit SparsificationNode(float sparsity)
      : _compiled(false), _sparsity(sparsity) {}

 public:
  static auto make(float sparsity) {
    return std::shared_ptr<SparsificationNode>(
        new SparsificationNode(sparsity));
  }

  auto addPredecessor(NodePtr input) {
    if (getState() != NodeState::Constructed) {
      throw exceptions::NodeStateMachineError(
          "SparsificationNode expected to have exactly one predecessor, and "
          "addPredecessor cannot be called twice.");
    }

    _input = std::move(input);

    return shared_from_this();
  }

  uint32_t outputDim() const final { return _input->outputDim(); }

  bool isInputNode() const final { return false; }

  void initOptimizer() final {}

  bool hasParameters() final { return false; }

 private:
  void compileImpl() final { _compiled = true; }

  std::vector<std::shared_ptr<FullyConnectedLayer>>
  getInternalFullyConnectedLayersImpl() const final {
    return {};
  }

  void prepareForBatchProcessingImpl(uint32_t batch_size,
                                     bool use_sparsity) final {
    if (_input->numNonzerosInOutput() != _input->outputDim()) {
      throw std::runtime_error(
          "Sparsification op can only be applied to dense vectors.");
    }
    uint32_t dim =
        use_sparsity ? _sparsity * _input->outputDim() : _input->outputDim();

    _outputs = BoltBatch(/* dim= */ dim, /* batch_size= */ batch_size,
                         /* is_dense= */ !use_sparsity);
  }

  uint32_t numNonzerosInOutputImpl() const final { return (*_outputs)[0].len; }

  void forwardImpl(uint32_t vec_index, const BoltVector* labels) final;

  void backpropagateImpl(uint32_t vec_index) final;

  void updateParametersImpl(float learning_rate, uint32_t batch_cnt) final {
    (void)learning_rate;
    (void)batch_cnt;
  }

  BoltVector& getOutputVectorImpl(uint32_t vec_index) final {
    return (*_outputs)[vec_index];
  }

  std::vector<NodePtr> getPredecessorsImpl() const final { return {_input}; }

  void summarizeImpl(std::stringstream& summary, bool detailed) const final {
    (void)detailed;
    summary << _input->name() << " -> " << name()
            << " (Sparsification): sparsity=" << _sparsity << std::endl;
  }

  std::string type() const final { return "sparsification"; }

  NodeState getState() const final;

  bool _compiled;
  NodePtr _input;
  float _sparsity;
  std::optional<BoltBatch> _outputs;

  SparsificationNode() : _compiled(false) {}

  friend class cereal::access;
  template <class Archive>
  void serialize(Archive& archive);
};

using SparsificationNodePtr = std::shared_ptr<SparsificationNode>;

}  // namespace thirdai::bolt