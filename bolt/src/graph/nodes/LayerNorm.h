#pragma once

#include <cereal/archives/binary.hpp>
#include <cereal/types/base_class.hpp>
#include <cereal/types/memory.hpp>
#include <cereal/types/optional.hpp>
#include <bolt/src/graph/Node.h>
#include <bolt/src/layers/LayerConfig.h>
#include <bolt_vector/src/BoltVector.h>
#include <exceptions/src/Exceptions.h>
#include <cmath>
#include <memory>
#include <optional>
#include <stdexcept>
#include <utility>

namespace thirdai::bolt {

/*
  This class normalizes the activations of the previous layer for each sample
  independently rather than applying a normalization across a batch of samples.
  The transformation sets the activations' means and variances close to 0 and
  1 respectively.
  If the `scale` and `center` parameters are set, the corresponding z-scores
  are linearly transformed according to the parameters specified by the beta and
  gamma regularizers.
 */

// Fixed constant to guard against dividing by zero in the
// gradient computation in the rare cases when the standard deviation
// evaluates to zero.
constexpr float DIV_BY_ZERO_GUARD = 0.00000001;

class LayerNormNode final : public Node,
                            public std::enable_shared_from_this<LayerNormNode> {
 private:
  LayerNormNode();

  explicit LayerNormNode(const NormalizationLayerConfig& config);

 public:
  static std::shared_ptr<LayerNormNode> make() {
    return std::shared_ptr<LayerNormNode>(new LayerNormNode());
  }

  static std::shared_ptr<LayerNormNode> makeWithConfig(
      const NormalizationLayerConfig& config) {
    return std::shared_ptr<LayerNormNode>(new LayerNormNode(config));
  }

  std::shared_ptr<LayerNormNode> addPredecessor(NodePtr node);

  uint32_t outputDim() const final { return _node_to_normalize->outputDim(); }

  bool isInputNode() const final {
    // This should not be an input node
    return false;
  }

  void initOptimizer() final {}

  bool hasParameters() final { return false; }

 private:
  void compileImpl() final { _compiled = true; }

  std::vector<std::shared_ptr<FullyConnectedLayer>>
  getInternalFullyConnectedLayersImpl() const final {
    return {};
  }

  void prepareForBatchProcessingImpl(uint32_t batch_size,
                                     bool use_sparsity) final;

  // Computes the first and second moments {mean, variance} required
  // to normalize the input to this layer.
  static std::tuple<float, float> computeNormalizationMoments(
      const BoltVector& bolt_vector);

  void forwardImpl(uint32_t vec_index, const BoltVector* labels) final;

  // Computes the derivative of the normalization function
  // For activation x, the normalization is given by
  // f(x) = [(x - mu)/(sigma + epsilon)] * gamma + beta
  // For a layer with n activations, the expression for the partial derivative
  // can be found here
  // https://www.notion.so/Bolt-DAG-API-Proposal-8d2d72d13df94f64b7829f80ab080def#0d4ec531c9f64e83a460bd56dfe04320
  float normDerivative(float activation, float mean, float variance,
                       uint32_t vec_length);

  void backpropagateImpl(uint32_t vec_index) final;

  void updateParametersImpl(float learning_rate, uint32_t batch_cnt) final;

  BoltVector& getOutputVectorImpl(uint32_t vec_index) final {
    return (*_batch)[vec_index];
  }

  uint32_t numNonzerosInOutputImpl() const final {
    return _node_to_normalize->numNonzerosInOutput();
  }

  std::vector<NodePtr> getPredecessorsImpl() const final {
    return {_node_to_normalize};
  }

  void summarizeImpl(std::stringstream& summary, bool detailed) const final;

  std::string type() const final { return std::string("layer_norm"); }

  NodeState getState() const final;

  friend class cereal::access;
  template <class Archive>
  void serialize(Archive& archive) {
    archive(cereal::base_class<Node>(this), _config, _node_to_normalize,
            _compiled);
  }

  std::shared_ptr<NormalizationLayerConfig> _config;

  // This private field is std::nullopt until after the node enters the
  // prepared for batch processing state
  std::optional<BoltBatch> _batch;
  NodePtr _node_to_normalize;
  bool _compiled;
};

}  // namespace thirdai::bolt
