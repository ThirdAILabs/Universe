#pragma once

#include <bolt/src/graph/Node.h>
#include <bolt_vector/src/BoltVector.h>
#include <memory>

namespace thirdai::bolt {

class DotProductNode final
    : public Node,
      public std::enable_shared_from_this<DotProductNode> {
 private:
  DotProductNode() : _compiled(false) {}

 public:
  static std::shared_ptr<DotProductNode> make() {
    return std::shared_ptr<DotProductNode>(new DotProductNode());
  }

  uint32_t outputDim() const final { return 1; }

  bool isInputNode() const final { return false; }

  void initOptimizer() final {}

  std::shared_ptr<DotProductNode> setPredecessors(NodePtr lhs, NodePtr rhs) {
    _lhs = std::move(lhs);
    _rhs = std::move(rhs);

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
    _outputs = BoltBatch(/* dim= */ 1, /* batch_size=*/batch_size,
                         /* is_dense= */ true);
  }

  uint32_t numNonzerosInOutputImpl() const final { throw 1; }

  void forwardImpl(uint32_t vec_index, const BoltVector* labels) final {
    (void)labels;

    BoltVector& a = _lhs->getOutputVector(vec_index);
    BoltVector& b = _rhs->getOutputVector(vec_index);

    float output;
    if (a.isDense()) {
      if (b.isDense()) {
        output = denseDenseDotProduct(a, b);
      } else {
        output = denseSparseDotProduct(a, b);
      }
    } else {
      if (b.isDense()) {
        output = denseSparseDotProduct(b, a);
      } else {
        output = sparseSparseDotProduct(a, b);
      }
    }

    (*_outputs)[vec_index].activations[0] = 1 / (1 + std::exp(-output));
  }

  void backpropagateImpl(uint32_t vec_index) final {
    BoltVector& a = _lhs->getOutputVector(vec_index);
    BoltVector& b = _rhs->getOutputVector(vec_index);

    float grad = (*_outputs)[vec_index].gradients[0];

    if (a.isDense()) {
      if (b.isDense()) {
        denseDenseDotProductBackward(grad, a, b);
      } else {
        denseSparseDotProductBackward(grad, a, b);
      }
    } else {
      if (b.isDense()) {
        denseSparseDotProductBackward(grad, b, a);
      } else {
        sparseSparseDotProductBackward(grad, a, b);
      }
    }
  }

  void updateParametersImpl(float learning_rate, uint32_t batch_cnt) final {
    (void)learning_rate;
    (void)batch_cnt;
  }

  BoltVector& getOutputVectorImpl(uint32_t vec_index) final {
    return (*_outputs)[vec_index];
  }

  void cleanupAfterBatchProcessingImpl() final { _outputs = std::nullopt; }

  void summarizeImpl(std::stringstream& summary, bool detailed) const final {
    (void)detailed;
    summary << "(" << _lhs->name() << ", " << _rhs->name() << ") -> " << name()
            << " (DotProduct)\n";
  }

  void enableDistributedTraining() final {
    // NOOP since the DotProduct node doesn't have any paramters
  }

  std::string type() const final { return "dot_product"; }

  std::vector<NodePtr> getPredecessorsImpl() const final {
    return {_lhs, _rhs};
  }

  NodeState getState() const final {
    if ((!_lhs && !_rhs) && !_compiled && !_outputs) {
      return NodeState::Constructed;
    }
    if ((_lhs && _rhs) && !_compiled && !_outputs) {
      return NodeState::PredecessorsSet;
    }
    if ((_lhs && _rhs) && _compiled && !_outputs) {
      return NodeState::Compiled;
    }
    if ((_lhs && _rhs) && _compiled && _outputs) {
      return NodeState::PreparedForBatchProcessing;
    }
    throw exceptions::NodeStateMachineError(
        "InputNode is in an invalid internal state");
  }

 private:
  static float denseDenseDotProduct(const BoltVector& a, const BoltVector& b) {
    assert(a.len == b.len);
    assert(a.isDense() && b.isDense());

    float total = 0;
    for (uint32_t i = 0; i < a.len; i++) {
      total += a.activations[i] * b.activations[i];
    }
    return total;
  }

  static void denseDenseDotProductBackward(float grad, const BoltVector& a,
                                           const BoltVector& b) {
    assert(a.len == b.len);
    assert(a.isDense() && b.isDense());

    for (uint32_t i = 0; i < a.len; i++) {
      a.gradients[i] += grad * b.activations[i];
      b.gradients[i] += grad * a.activations[i];
    }
  }

  static float denseSparseDotProduct(const BoltVector& dense_vec,
                                     const BoltVector& sparse_vec) {
    assert(dense_vec.isDense() && !sparse_vec.isDense());

    float total = 0;
    for (uint32_t i = 0; i < sparse_vec.len; i++) {
      assert(sparse_vec.active_neurons[i] < dense_vec.len);
      total += dense_vec.activations[sparse_vec.active_neurons[i]] *
               sparse_vec.activations[i];
    }
    return total;
  }

  static void denseSparseDotProductBackward(float grad,
                                            const BoltVector& dense_vec,
                                            const BoltVector& sparse_vec) {
    assert(dense_vec.isDense() && !sparse_vec.isDense());

    for (uint32_t i = 0; i < sparse_vec.len; i++) {
      assert(sparse_vec.active_neurons[i] < dense_vec.len);

      uint32_t active_neuron = sparse_vec.active_neurons[i];
      sparse_vec.gradients[i] += grad * dense_vec.activations[active_neuron];
      dense_vec.gradients[active_neuron] += grad * sparse_vec.activations[i];
    }
  }

  static float sparseSparseDotProduct(BoltVector& a, BoltVector& b) {
    assert(!a.isDense() && !b.isDense());

    a.sortActiveNeurons();
    b.sortActiveNeurons();

    float total = 0.0;

    uint32_t a_index = 0;
    uint32_t b_index = 0;
    while (a_index < a.len && b_index < b.len) {
      if (a.active_neurons[a_index] == b.active_neurons[b_index]) {
        total += a.activations[a_index] * b.activations[b_index];
        a_index++;
        b_index++;
      } else if (a.active_neurons[a_index] < b.active_neurons[b_index]) {
        a_index++;
      } else {
        b_index++;
      }
    }

    return total;
  }

  static void sparseSparseDotProductBackward(float grad, BoltVector& a,
                                             BoltVector& b) {
    assert(!a.isDense() && !b.isDense());

    uint32_t a_index = 0;
    uint32_t b_index = 0;
    while (a_index < a.len && b_index < b.len) {
      if (a.active_neurons[a_index] == b.active_neurons[b_index]) {
        a.gradients[a_index] += grad * b.activations[b_index];
        b.gradients[b_index] += grad * a.activations[a_index];
        a_index++;
        b_index++;
      } else if (a.active_neurons[a_index] < b.active_neurons[b_index]) {
        a_index++;
      } else {
        b_index++;
      }
    }
  }

  NodePtr _lhs;
  NodePtr _rhs;
  bool _compiled;

  std::optional<BoltBatch> _outputs;
};

using DotProductNodePtr = std::shared_ptr<DotProductNode>;

}  // namespace thirdai::bolt