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
      : _moments(std::nullopt),
        _predecessor(nullptr),
        _compiled(false),
        _prepared_for_batch_processing(false) {}

  std::shared_ptr<LayerNormNode> addPredecessor(NodePtr node) {
    if (getState() != NodeState::Constructed) {
      throw exceptions::NodeStateMachineError(
          "LayerNormNode should have exactly one predecessor."
          "addPredecessor cannot be called twice.");
    }
    _predecessor = std::move(node);
    return shared_from_this();
  }

  std::shared_ptr<LayerNormNode> setLayerNormNodeConfig(
      bool center, bool scale, float epsilon, float beta_regularizer,
      float gamma_regularizer, float beta_initializer,
      float gamma_initializer) {
    if (getState() != NodeState::Constructed) {
      throw exceptions::NodeStateMachineError(
          "Cannot set configuration for Normalization Layer before the node is "
          "Constructed.");
    }

    _config = (NormalizationLayerConfig(
        /* beta_regularizer */ beta_regularizer,
        /* gamma_regularizer */ gamma_regularizer,
        /* beta_initializer */ beta_initializer,
        /* gamma_initializer */ gamma_initializer,
        /* center */ center, /* scale */ scale, /* epsilon*/ epsilon));

    return shared_from_this();
  }

  uint32_t outputDim() const final { return _predecessor->outputDim(); }

  bool isInputNode() const final { return _predecessor->isInputNode(); }

  std::optional<std::pair<float, float>> getMoments() { return _moments; }

  static bool nodeIsSparse(const NodePtr& node) {
    return node->numNonzerosInOutput() < node->outputDim();
  }

  // Computes the first and second moments {mean, variance} required
  // to normalize the input to this layer.
  static std::pair<float, float> computeNormalizationMoments(
      const BoltVector& bolt_vector, bool is_dense) {
    uint32_t len = bolt_vector.len;
    float mean = 0, variance = 0;

    for (uint32_t neuron_index = 0; neuron_index < len; neuron_index++) {
      auto active_neuron =
          is_dense ? bolt_vector.findActiveNeuron<true>(neuron_index)
                   : bolt_vector.findActiveNeuron<false>(neuron_index);

      mean += active_neuron.activation;
    }
    mean /= len;
    for (uint32_t neuron_index = 0; neuron_index < len; neuron_index++) {
      auto active_neuron =
          is_dense ? bolt_vector.findActiveNeuron<true>(neuron_index)
                   : bolt_vector.findActiveNeuron<false>(neuron_index);

      variance += pow((active_neuron.activation - mean), 2.0);
    }
    variance /= len;

    return std::make_pair(mean, variance);
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
    _predecessor->prepareForBatchProcessing(batch_size, use_sparsity);
    bool is_sparse = nodeIsSparse(_predecessor);

    if (is_sparse && !use_sparsity) {
      throw exceptions::NodeStateMachineError(
          "Input to the Layer Normalization is a sparse vector, but "
          "use_sparsity "
          "in the call to prepareForBatchProcessing is set to False");
    }
    _prepared_for_batch_processing = true;
  }

  void forwardImpl(uint32_t vec_index, const BoltVector* labels) final {
    // Assumes that layer normalization is not applied to the last layer
    assert(labels == nullptr);

    (void)labels;

    const BoltVector& output_vector = getOutputVectorImpl(vec_index);
    bool is_sparse = nodeIsSparse(_predecessor);
    std::vector<float> normalized_activations = {0};

    uint32_t len = output_vector.len;

    assert(len != 0);
    _moments = computeNormalizationMoments(output_vector, !is_sparse);

    for (uint32_t neuron_index = 0; neuron_index < len; neuron_index++) {
      auto active_neuron =
          is_sparse ? output_vector.findActiveNeuron<false>(neuron_index)
                    : output_vector.findActiveNeuron<true>(neuron_index);

      auto z_score = (active_neuron.activation - _moments->first) /
                     sqrt(_moments->second + _config->epsilon);

      // apply a linear transformation to the z_score using gamma and beta
      // regularizers
      z_score += (_config->center) ? _config->beta_regularizer : 0;
      z_score *= (_config->scale) ? _config->gamma_regularizer : 1;
      normalized_activations.push_back(z_score);
    }

    std::copy(normalized_activations.begin(), normalized_activations.end(),
              output_vector.activations);
  }

  void backpropagateImpl(uint32_t vec_index) final {
    // Layer normalization does not change the gradients. Thus, no gradient
    // updates are required.
    (void)vec_index;
  }

  void updateParametersImpl(float learning_rate, uint32_t batch_cnt) final {
    (void)learning_rate;
    (void)batch_cnt;
  }

  BoltVector& getOutputVectorImpl(uint32_t vec_index) final {
    return _predecessor->getOutputVector(vec_index);
  }

  void cleanupAfterBatchProcessingImpl() final {
    _config = std::nullopt;
    _prepared_for_batch_processing = false;
  }

  uint32_t numNonzerosInOutputImpl() const final {
    // normalization does not alter the dimensions of the output, so we can
    // just return the result of the call to outputDim();
    return outputDim();
  }

  std::vector<NodePtr> getPredecessorsImpl() const final {
    return {_predecessor};
  }

  void summarizeImpl(std::stringstream& summary, bool detailed) const final {
    summary << _predecessor->name() << " -> " << name() << " (LayerNorm): ";
    (void)detailed;
  }

  std::string type() const final { return std::string("layer_norm"); }

  NodeState getState() const final {
    if (!_predecessor && !_compiled && !_prepared_for_batch_processing) {
      return NodeState::Constructed;
    }
    if (_predecessor && !_compiled && !_prepared_for_batch_processing) {
      return NodeState::PredecessorsSet;
    }
    if (_predecessor && _compiled && !_prepared_for_batch_processing) {
      return NodeState::Compiled;
    }
    if (_predecessor && _compiled && _prepared_for_batch_processing) {
      return NodeState::PreparedForBatchProcessing;
    }
    throw exceptions::NodeStateMachineError(
        "LayerNormNode is in an invalid internal state");
  }

  std::optional<NormalizationLayerConfig> _config;
  std::optional<std::pair<float, float>> _moments;
  NodePtr _predecessor;
  bool _compiled;
  bool _prepared_for_batch_processing;
};

}  // namespace thirdai::bolt