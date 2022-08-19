#pragma once

#include <cereal/archives/binary.hpp>
#include <cereal/types/base_class.hpp>
#include <cereal/types/memory.hpp>
#include <cereal/types/optional.hpp>
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
  are linearly transformed according to the parameters specified by the beta and
  gamma regularizers.
 */

// Fixed constant to guard against dividing by zero in the
// gradient computation in the rare cases when the standard deviation
// evaluates to zero.
constexpr float DIV_BY_ZERO_GUARD = 0.00000001;

class LayerNormNode final : public Node,
                            public std::enable_shared_from_this<LayerNormNode> {
 public:
  LayerNormNode()
      : _config(std::make_shared<NormalizationLayerConfig>()),
        _batch(std::nullopt),
        _node_to_normalize(nullptr),
        _compiled(false) {}

  explicit LayerNormNode(const NormalizationLayerConfig& config)
      : _config(std::make_shared<NormalizationLayerConfig>(config)),
        _batch(std::nullopt),
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

  void initTrainDatastructures() final {}

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

    auto dim = _node_to_normalize->numNonzerosInOutput();

    _batch = BoltBatch(/* dim=*/dim, /* batch_size= */ batch_size,
                       /* is_dense= */ is_dense);
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
    (void)labels;

    const BoltVector& input_vector =
        _node_to_normalize->getOutputVector(vec_index);

    auto output = getOutputVectorImpl(vec_index);

    auto [mean, variance] = computeNormalizationMoments(input_vector);

    assert(!std::isnan(mean));
    assert(!std::isnan(variance));

    for (uint32_t neuron_index = 0; neuron_index < input_vector.len;
         neuron_index++) {
      float activation = input_vector.activations[neuron_index];

      // The epsilon factor is to guard against division by zero.
      auto z_score =
          (activation - mean) / (sqrt(variance) + _config->epsilon());

      // apply a linear transformation to the z_score using gamma and beta
      // regularizers.
      z_score += _config->beta();
      z_score *= _config->gamma();
      (*_batch)[vec_index].activations[neuron_index] = z_score;
    }
  }

  // Computes the derivative of the normalization function
  // For activation x, the normalization is given by
  // f(x) = [(x - mu)/(sigma + epsilon)] * gamma + beta
  // For a layer with n activations, the expression for the partial derivative
  // can be found here
  // https://www.notion.so/Bolt-DAG-API-Proposal-8d2d72d13df94f64b7829f80ab080def#0d4ec531c9f64e83a460bd56dfe04320

  float normDerivative(float activation, float mean, float variance,
                       uint32_t vec_length) {
    assert(getState() == NodeState::PreparedForBatchProcessing);

    float centered_activation = (activation - mean) * (activation - mean);
    float std_deviation = sqrt(variance);
    auto denominator = (vec_length * vec_length) * std_deviation *
                       (std_deviation + _config->epsilon()) *
                       (std_deviation + _config->epsilon());

    // additive term to avoid division by zero
    denominator += DIV_BY_ZERO_GUARD;

    auto gradient =
        vec_length * std_deviation * (std_deviation + _config->epsilon()) -
        centered_activation;
    gradient /= denominator;
    gradient *= _config->gamma() * (vec_length - 1);

    return gradient;
  }

  void backpropagateImpl(uint32_t vec_index) final {
    BoltVector& input_vector = _node_to_normalize->getOutputVector(vec_index);
    BoltVector& output_vector = getOutputVectorImpl(vec_index);

    auto [mean, variance] = computeNormalizationMoments(input_vector);

    uint32_t len = input_vector.len;

    for (uint32_t neuron_index = 0; neuron_index < input_vector.len;
         neuron_index++) {
      float grad = normDerivative(input_vector.activations[neuron_index], mean,
                                  variance, len);

      assert(!std::isnan(grad));

      assert(!std::isnan(output_vector.gradients[neuron_index]));

      if (grad == 0.0 || output_vector.gradients[neuron_index] == 0) {
        continue;
      }

      input_vector.gradients[neuron_index] =
          output_vector.gradients[neuron_index] * grad;
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
    return (*_batch)[vec_index];
  }

  void cleanupAfterBatchProcessingImpl() final { _batch = std::nullopt; }

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
      summary << ", epsilon=" << _config->epsilon()
              << ", beta_regularizer=" << _config->beta();
      summary << ", gamma_regularizer=" << _config->gamma();
      summary << ")";
    }
    summary << "\n";
  }

  std::string type() const final { return std::string("layer_norm"); }

  NodeState getState() const final {
    if (!_node_to_normalize && !_compiled && !_batch) {
      return NodeState::Constructed;
    }
    if (_node_to_normalize && !_compiled && !_batch) {
      return NodeState::PredecessorsSet;
    }
    if (_node_to_normalize && _compiled && !_batch) {
      return NodeState::Compiled;
    }
    if (_node_to_normalize && _compiled && _batch) {
      return NodeState::PreparedForBatchProcessing;
    }
    throw exceptions::NodeStateMachineError(
        "LayerNormNode is in an invalid internal state");
  }

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

CEREAL_REGISTER_TYPE(thirdai::bolt::LayerNormNode)