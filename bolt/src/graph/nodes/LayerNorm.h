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
    // This should not be an input node
    return false;
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

    BoltBatch batch =
        BoltBatch(/* dim=*/outputDim(), /* batch_size= */ batch_size,
                  /* is_dense= */ is_dense);

    _layer_norm_state = LayerNormState(batch);
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

      variance += (activation - mean) * (activation - mean);
    }
    variance /= len;

    return std::make_pair(mean, variance);
  }

  void forwardImpl(uint32_t vec_index, const BoltVector* labels) final {
    // Assumes that layer normalization is not applied to the last layer
    assert(labels == nullptr);

    (void)labels;

    const BoltVector& input_vector =
        _node_to_normalize->getOutputVector(vec_index);

    auto [mean, variance] = computeNormalizationMoments(input_vector);

    for (uint32_t neuron_index = 0; neuron_index < input_vector.len;
         neuron_index++) {
      float activation = input_vector.activations[neuron_index];

      auto z_score = (activation - mean) / sqrt(variance + _config->epsilon());

      // apply a linear transformation to the z_score using gamma and beta
      // regularizers
      z_score += (_config->center()) ? _config->beta() : 0;
      z_score *= (_config->scale()) ? _config->gamma() : 1;
      _layer_norm_state->outputs[vec_index].activations[neuron_index] = z_score;
    }
  }

  // Computes the derivative of the normalization function
  // For activation x, the normalization is given by
  // f(x) = [(x - mu)/(sigma + epsilon)] * gamma + beta
  // For a layer with n activations, the partial derivative is expressed by
  // gamma * (n-1) * [(n*sigma^2) - [(x-mu)^2]] / (n^2* sigma^3)
  float normDerivative(float activation, float mean, float variance,
                       uint32_t vec_length) {
    assert(getState() == NodeState::PreparedForBatchProcessing);

    float centered_activation = (activation - mean) * (activation - mean);
    auto denominator = (vec_length * vec_length) * variance * sqrt(variance);

    auto gradient =
        ((vec_length - 1) * (variance * vec_length - centered_activation)) /
        (denominator + _config->epsilon());
    gradient *= _config->gamma();

    return gradient;
  }

  void backpropagateImpl(uint32_t vec_index) final {
    BoltVector& input_vector = _node_to_normalize->getOutputVector(vec_index);
    BoltVector& output_vector = getOutputVectorImpl(vec_index);

    auto [mean, variance] = computeNormalizationMoments(input_vector);

    uint32_t len = input_vector.len;

    for (uint32_t neuron_index = 0; neuron_index < input_vector.len;
         neuron_index++) {
      auto output_vector_activation = output_vector.activations[neuron_index];
      float grad =
          normDerivative(output_vector_activation, mean, variance, len);

      assert(!std::isnan(grad));
      output_vector.gradients[neuron_index] = grad;
      input_vector.gradients[neuron_index] +=
          sqrt(variance) / (_config->gamma() + _config->epsilon());
    }
  }

  void updateParametersImpl(float learning_rate, uint32_t batch_cnt) final {
    // TODO(blaise): Since _gamma_regularizer and _beta_regularizer are
    // trainable parameters, we should add an implementation for updating these
    // parameters
    (void)learning_rate;
    (void)batch_cnt;
  }

  BoltVector& getOutputVectorImpl(uint32_t vec_index) final {
    assert(getState() == NodeState::PreparedForBatchProcessing);
    return (_layer_norm_state->outputs)[vec_index];
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
    explicit LayerNormState(BoltBatch& batch) : outputs(std::move(batch)) {}

    BoltBatch outputs;
  };

  friend class cereal::access;
  template <class Archive>
  void serialize(Archive& archive) {
    archive(cereal::base_class<Node>(this), _config, _node_to_normalize,
            _compiled);
  }

  std::shared_ptr<NormalizationLayerConfig> _config;

  // This private field is std::nullopt until after the node enters the
  // prepared for batch normalization state
  std::optional<LayerNormState> _layer_norm_state;
  NodePtr _node_to_normalize;
  bool _compiled;
};

}  // namespace thirdai::bolt