#pragma once

#include <cereal/access.hpp>
#include <cereal/types/base_class.hpp>
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

  void saveWithOptimizer(bool should_save_optimizer) final {
    (void)should_save_optimizer;
  };

  std::shared_ptr<DotProductNode> setPredecessors(NodePtr lhs, NodePtr rhs);

  bool hasParameters() final { return false; }

 protected:
  void compileImpl() final { _compiled = true; }

  std::vector<std::shared_ptr<FullyConnectedLayer>>
  getInternalFullyConnectedLayersImpl() const final {
    return {};
  }

  void prepareForBatchProcessingImpl(uint32_t batch_size,
                                     bool use_sparsity) final;

  uint32_t numNonzerosInOutputImpl() const final { return 1; }

  void forwardImpl(uint32_t vec_index, const BoltVector* labels) final;

  void backpropagateImpl(uint32_t vec_index) final;

  void updateParametersImpl(float learning_rate, uint32_t batch_cnt) final {
    (void)learning_rate;
    (void)batch_cnt;
  }

  BoltVector& getOutputVectorImpl(uint32_t vec_index) final {
    return (*_outputs)[vec_index];
  }

  void summarizeImpl(std::stringstream& summary, bool detailed) const final;

  std::string type() const final { return "dot_product"; }

  std::vector<NodePtr> getPredecessorsImpl() const final {
    return {_lhs, _rhs};
  }

  NodeState getState() const final;

 private:
  static float denseDenseDotProduct(const BoltVector& a, const BoltVector& b);

  static void denseDenseDotProductBackward(float grad, const BoltVector& a,
                                           const BoltVector& b);

  static float denseSparseDotProduct(const BoltVector& dense_vec,
                                     const BoltVector& sparse_vec);

  static void denseSparseDotProductBackward(float grad,
                                            const BoltVector& dense_vec,
                                            const BoltVector& sparse_vec);

  static float sparseSparseDotProduct(BoltVector& a, BoltVector& b);

  static void sparseSparseDotProductBackward(float grad, BoltVector& a,
                                             BoltVector& b);

  static void applyFunctionToOverlappingNeurons(
      BoltVector& a, BoltVector& b,
      const std::function<void(uint32_t, uint32_t)>& func);

  friend class cereal::access;
  template <class Archive>
  void serialize(Archive& archive) {
    archive(cereal::base_class<Node>(this), _lhs, _rhs, _compiled);
  }

  NodePtr _lhs;
  NodePtr _rhs;
  bool _compiled;

  std::optional<BoltBatch> _outputs;
};

using DotProductNodePtr = std::shared_ptr<DotProductNode>;

}  // namespace thirdai::bolt
