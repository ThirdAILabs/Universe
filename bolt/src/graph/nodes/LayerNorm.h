#pragma once

#include <bolt/src/graph/Node.h>
#include <bolt/src/layers/BoltVector.h>
#include <bolt/src/layers/LayerConfig.h>
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
  are linearly transformed according to the parameters specified by beta and
  gamma.
 */

class LayerNormNode final : public Node,
                            public std::enable_shared_from_this<LayerNormNode> {
 public:
  LayerNormNode()
      : _config(std::make_shared<NormalizationLayerConfig>()),
        _layer_norm_state(std::nullopt),
        _node_to_normalize(nullptr),
        _compiled(false) {}

  explicit LayerNormNode(const NormalizationLayerConfig& config)
      : _config(std::make_shared<NormalizationLayerConfig>(config)),
        _layer_norm_state(std::nullopt),
        _node_to_normalize(nullptr),
        _compiled(false) {}

  std::shared_ptr<LayerNormNode> addPredecessor(NodePtr node) {
    if (getState() != NodeState::Constructed) {
      throw exceptions::NodeStateMachineError(
          "Predecessor Node has already been set for this "
          "Normalization Layer. ");
    }
    assert(!node->isInputNode());

    _node_to_normalize = std::move(node);
    return shared_from_this();
  }

  uint32_t outputDim() const final { return _node_to_normalize->outputDim(); }

  bool isInputNode() const final {
    // _node_to_normalize should not be an input node
    return false;
  }

  // Computes the first and second moments {mean, variance} required
  // to normalize the input to this layer.
  static std::pair<float, float> computeNormalizationMoments(
      const BoltVector& bolt_vector) {
    uint32_t len = bolt_vector.len;
    float mean = 0, variance = 0;

    for (uint32_t neuron_index = 0; neuron_index < len; neuron_index++) {
      mean += bolt_vector.activations[neuron_index];
    }
    mean /= len;
    for (uint32_t neuron_index = 0; neuron_index < len; neuron_index++) {
      float activation = bolt_vector.activations[neuron_index];

      variance += pow((activation - mean), 2.0);
    }
    variance /= len;

    return std::make_pair(mean, variance);
  }

  std::optional<std::pair<float, float>> getMoments(uint32_t vec_index) {
    if (!_layer_norm_state.has_value()) {
      return std::nullopt;
    }
    return _layer_norm_state->moments[vec_index];
  }

 private:
  void compileImpl() final { _compiled = true; }

  std::vector<std::shared_ptr<FullyConnectedLayer>>
  getInternalFullyConnectedLayersImpl() const final {
    return {};
  }

  void prepareForBatchProcessingImpl(uint32_t batch_size,
                                     bool use_sparsity) final {
    (void)use_sparsity;
    bool is_dense = _node_to_normalize->numNonzerosInOutput() ==
                    _node_to_normalize->outputDim();

    BoltBatch batch = BoltBatch(outputDim(), batch_size, is_dense);
    std::vector<std::pair<float, float>> moments(batch_size,
                                                 std::make_pair(0.0, 0.0));

    _layer_norm_state = LayerNormState(/* outputs = */ std::move(batch),
                                       /* moments = */ std::move(moments));

    for (uint32_t index_in_batch = 0; index_in_batch < batch_size;
         index_in_batch++) {
      const BoltVector& output_vector = getOutputVectorImpl(index_in_batch);

      assert(output_vector.len != 0);
      auto moments = computeNormalizationMoments(output_vector);

      _layer_norm_state->moments.push_back(moments);
    }
  }

  void forwardImpl(uint32_t vec_index, const BoltVector* labels) final {
    // Assumes that layer normalization is not applied to the last layer
    assert(labels == nullptr);

    (void)labels;

    const BoltVector& output_vector = getOutputVectorImpl(vec_index);
    std::vector<float> normalized_activations = {0};

    auto moments = _layer_norm_state->moments[vec_index];

    for (uint32_t neuron_index = 0; neuron_index < output_vector.len;
         neuron_index++) {
      float activation = output_vector.activations[neuron_index];

      auto z_score = (activation - moments.first) /
                     sqrt(moments.second + _config->epsilon());

      // apply a linear transformation to the z_score using gamma and beta
      // regularizers
      z_score += (_config->center()) ? _config->beta() : 0;
      z_score *= (_config->scale()) ? _config->gamma() : 1;
      normalized_activations.push_back(z_score);
    }

    std::copy(normalized_activations.begin(), normalized_activations.end(),
              output_vector.activations);
  }

  // Computes the derivative of the normalization function
  float normDerivative(float activation, uint32_t vec_length,
                       uint32_t vec_index) {
    assert(getState() == NodeState::PreparedForBatchProcessing);

    float mean = _layer_norm_state->moments[vec_index].first;
    float variance = _layer_norm_state->moments[vec_index].second;

    float centered_activation = pow((activation - mean), 2.0);
    auto denominator = pow(vec_length, 2.0) * variance * sqrt(variance);

    auto gradient =
        ((vec_length - 1) * (variance * vec_length - centered_activation)) /
        denominator;
    gradient *= _config->gamma();

    return gradient;
  }

  void backpropagateImpl(uint32_t vec_index) final {
    BoltVector& output_vector = getOutputVectorImpl(vec_index);
    uint32_t len = output_vector.len;

    for (uint32_t neuron_index = 0; neuron_index < output_vector.len;
         neuron_index++) {
      auto activation = output_vector.activations[neuron_index];

      output_vector.gradients[neuron_index] =
          normDerivative(activation, len, vec_index);
    }
  }

  void updateParametersImpl(float learning_rate, uint32_t batch_cnt) final {
    (void)learning_rate;
    (void)batch_cnt;
  }

  BoltVector& getOutputVectorImpl(uint32_t vec_index) final {
    assert(getState() == NodeState::PreparedForBatchProcessing);
    return _layer_norm_state->outputs[vec_index];
  }

  void cleanupAfterBatchProcessingImpl() final {
    _layer_norm_state = std::nullopt;
  }

  uint32_t numNonzerosInOutputImpl() const final {
    return _node_to_normalize->numNonzerosInOutput();
  }

  std::vector<NodePtr> getPredecessorsImpl() const final {
    return {_node_to_normalize};
  }

  void summarizeImpl(std::stringstream& summary, bool detailed) const final {
    summary << _node_to_normalize->name() << " -> " << name()
            << " (LayerNorm) ";
    if (detailed) {
      summary << "(center=" << _config->center()
              << ", scale=" << _config->scale();
      summary << ", epsilon=" << _config->epsilon()
              << ", beta_regularizer=" << _config->beta();
      summary << ", gamma_regularizer=" << _config->gamma();
      summary << ")";
    }
    summary << "\n";
  }

  std::string type() const final { return std::string("layer_norm"); }

  NodeState getState() const final {
    if (!_node_to_normalize && !_compiled && !_layer_norm_state) {
      return NodeState::Constructed;
    }
    if (_node_to_normalize && !_compiled && !_layer_norm_state) {
      return NodeState::PredecessorsSet;
    }
    if (_node_to_normalize && _compiled && !_layer_norm_state) {
      return NodeState::Compiled;
    }
    if (_node_to_normalize && _compiled && _layer_norm_state) {
      return NodeState::PreparedForBatchProcessing;
    }
    throw exceptions::NodeStateMachineError(
        "LayerNormNode is in an invalid internal state");
  }

  struct LayerNormState {
    explicit LayerNormState(BoltBatch outputs,
                            std::vector<std::pair<float, float>> moments)
        : outputs(std::move(outputs)), moments(std::move(moments)) {}

    BoltBatch outputs;
    // The jth element in the vector corresponds to the (mean, variance) pair
    // for the jth vector in the batch
    std::vector<std::pair<float, float>> moments;
  };

  std::shared_ptr<NormalizationLayerConfig> _config;

  // This field will be optional except for when the node is in the batch
  // processing state
  std::optional<LayerNormState> _layer_norm_state;
  NodePtr _node_to_normalize;
  bool _compiled;
};

}  // namespace thirdai::bolt