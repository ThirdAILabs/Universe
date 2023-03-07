#pragma once

#include <bolt/src/graph/Node.h>
#include <bolt_vector/src/BoltVector.h>

namespace thirdai::bolt {

class Sparsification final : public Node {
 public:
  uint32_t outputDim() const final { return _input->outputDim(); }

  bool isInputNode() const final { return false; }

  void initOptimizer() final {}

  bool hasParameters() final { return false; }

  void compileImpl() final { _compiled = true; }

  std::vector<std::shared_ptr<FullyConnectedLayer>>
  getInternalFullyConnectedLayersImpl() const final {
    return {};
  }

  void prepareForBatchProcessingImpl(uint32_t batch_size,
                                     bool use_sparsity) final {
    uint32_t dim = use_sparsity ? _sparse_dim : _input->outputDim();

    _outputs = BoltBatch(dim, batch_size, dim == _sparse_dim);
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
            << " (Sparsification): sparse_dim=" << _sparse_dim;
  }

  std::string type() const final { return "sparsification"; }

  NodeState getState() const final;

 private:
  bool _compiled;

  NodePtr _input;

  uint32_t _sparse_dim;

  std::optional<BoltBatch> _outputs;
};

}  // namespace thirdai::bolt