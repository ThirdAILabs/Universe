#pragma once

#include <cereal/archives/binary.hpp>
#include <cereal/types/base_class.hpp>
#include <cereal/types/memory.hpp>
#include <cereal/types/optional.hpp>
#include <bolt/src/graph/Node.h>
#include <bolt/src/layers/LayerConfig.h>
#include <bolt/src/layers/LayerUtils.h>
#include <dataset/src/utils/SafeFileIO.h>
#include <exceptions/src/Exceptions.h>
#include <cstddef>
#include <memory>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <utility>

namespace thirdai::bolt {

class SwitchNode;

class FullyConnectedNode final
    : public Node,
      public std::enable_shared_from_this<FullyConnectedNode> {
  friend class SwitchNode;

 public:
  FullyConnectedNode(uint64_t dim, const std::string& activation)
      : _layer(nullptr),
        _config(FullyConnectedLayerConfig(dim, activation)),
        _predecessor(nullptr) {}

  FullyConnectedNode(uint64_t dim, float sparsity,
                     const std::string& activation)
      : _layer(nullptr),
        _config(FullyConnectedLayerConfig(dim, sparsity, activation)),
        _predecessor(nullptr) {}

  FullyConnectedNode(uint64_t dim, float sparsity,
                     const std::string& activation,
                     SamplingConfigPtr sampling_config)
      : _layer(nullptr),
        _config(FullyConnectedLayerConfig(dim, sparsity, activation,
                                          std::move(sampling_config))),
        _predecessor(nullptr) {}

  std::shared_ptr<FullyConnectedNode> addPredecessor(NodePtr node) {
    if (getState() != NodeState::Constructed) {
      throw exceptions::NodeStateMachineError(
          "FullyConnectedNode expected to have exactly one predecessor, and "
          "addPredecessor cannot be called twice.");
    }
    _predecessor = std::move(node);

    return shared_from_this();
  }

  uint32_t outputDim() const final {
    NodeState node_state = getState();
    if (node_state == NodeState::Constructed ||
        node_state == NodeState::PredecessorsSet) {
      return _config->getDim();
    }
    return _layer->getDim();
  }

  bool isInputNode() const final { return false; }

  ActivationFunction getActivationFunction() const {
    NodeState node_state = getState();
    if (node_state == NodeState::Constructed ||
        node_state == NodeState::PredecessorsSet) {
      return _config->getActFunc();
    }
    return _layer->getActivationFunction();
  }

  void saveParameters(const std::string& filename) const {
    std::ofstream filestream =
        dataset::SafeFileIO::ofstream(filename, std::ios::binary);
    cereal::BinaryOutputArchive oarchive(filestream);
    oarchive(*_layer);
  }

  void enableDistributedTraining() { _layer->enableDistributedTraining(); }

  void loadParameters(const std::string& filename) {
    std::ifstream filestream =
        dataset::SafeFileIO::ifstream(filename, std::ios::binary);
    cereal::BinaryInputArchive iarchive(filestream);
    auto loaded_parameters = std::make_shared<FullyConnectedLayer>();
    iarchive(*loaded_parameters);

    if (loaded_parameters->getDim() != _layer->getDim()) {
      std::stringstream error_msg;
      error_msg << "Cannot load parameters from FullyConnected layer with dim="
                << loaded_parameters->getDim()
                << "into FullyConnected layer with dim=" << _layer->getDim()
                << ".";
      throw std::logic_error(error_msg.str());
    }
    if (loaded_parameters->getInputDim() != _layer->getInputDim()) {
      std::stringstream error_msg;
      error_msg
          << "Cannot load parameters from FullyConnected layer with input_dim="
          << loaded_parameters->getInputDim()
          << "into FullyConnected layer with input_dim="
          << _layer->getInputDim() << ".";
      throw std::logic_error(error_msg.str());
    }

    if (loaded_parameters->getActivationFunction() !=
        _layer->getActivationFunction()) {
      std::stringstream error_msg;
      error_msg << "Cannot load parameters from FullyConnected layer with "
                   "activation ="
                << activationFunctionToStr(
                       loaded_parameters->getActivationFunction())
                << "into FullyConnected layer with activation="
                << activationFunctionToStr(_layer->getActivationFunction())
                << ".";
      throw std::logic_error(error_msg.str());
    }

    _layer = loaded_parameters;
  }

  float getSparsity() {
    NodeState node_state = getState();
    if (node_state == NodeState::Constructed ||
        node_state == NodeState::PredecessorsSet) {
      return _config->getSparsity();
    }
    return _layer->getSparsity();
  }

  std::shared_ptr<FullyConnectedNode> setSparsity(float sparsity) {
    if (getState() != NodeState::Compiled) {
      throw exceptions::NodeStateMachineError(
          "FullyConnectedNode must be in a compiled state to call setSparsity");
    }
    _layer->setSparsity(sparsity);
    return shared_from_this();
  }

  float* getWeightsPtr() {
    if (getState() != NodeState::PreparedForBatchProcessing &&
        getState() != NodeState::Compiled) {
      throw exceptions::NodeStateMachineError(
          "FullyConnectedNode must be in a compiled state to call "
          "getWeightsPtr.");
    }
    return _layer->getWeightsPtr();
  }

  float* getBiasesPtr() {
    if (getState() != NodeState::PreparedForBatchProcessing &&
        getState() != NodeState::Compiled) {
      throw exceptions::NodeStateMachineError(
          "FullyConnectedNode must be in a compiled state to call "
          "getBiasesPtr.");
    }
    return _layer->getBiasesPtr();
  }

  float* getWeightGradientsPtr() {
    if (getState() != NodeState::PreparedForBatchProcessing &&
        getState() != NodeState::Compiled) {
      throw exceptions::NodeStateMachineError(
          "FullyConnectedNode must be in a compiled state to call "
          "getWeightGradientsPtr.");
    }
    return _layer->getWeightGradientsPtr();
  }

  float* getBiasGradientsPtr() {
    if (getState() != NodeState::PreparedForBatchProcessing &&
        getState() != NodeState::Compiled) {
      throw exceptions::NodeStateMachineError(
          "FullyConnectedNode must be in a compiled state to call "
          "getBiasGradientsPtr.");
    }
    return _layer->getBiasGradientsPtr();
  }

  static std::shared_ptr<FullyConnectedNode> make(uint32_t dim,
                                                  std::string activation) {
    return std::make_shared<FullyConnectedNode>(dim, std::move(activation));
  }

  static std::shared_ptr<FullyConnectedNode> make(uint32_t dim, float sparsity,
                                                  std::string activation) {
    return std::make_shared<FullyConnectedNode>(dim, sparsity,
                                                std::move(activation));
  }

  static std::shared_ptr<FullyConnectedNode> make(uint32_t dim, float sparsity,
                                                  std::string activation,
                                                  uint32_t num_tables,
                                                  uint32_t hashes_per_table,
                                                  uint32_t reservoir_size) {
    auto sampling_config = std::make_shared<DWTASamplingConfig>(
        num_tables, hashes_per_table, reservoir_size);
    return std::make_shared<FullyConnectedNode>(dim, sparsity, activation,
                                                sampling_config);
  }

 private:
  void compileImpl() final {
    assert(_config.has_value());
    _layer = std::make_shared<FullyConnectedLayer>(_config.value(),
                                                   _predecessor->outputDim());
    _config = std::nullopt;
  }

  std::vector<std::shared_ptr<FullyConnectedLayer>>
  getInternalFullyConnectedLayersImpl() const final {
    return {_layer};
  }

  void prepareForBatchProcessingImpl(uint32_t batch_size,
                                     bool use_sparsity) final {
    // TODO(Nicholas): rename createBatchState
    _outputs =
        _layer->createBatchState(batch_size, /* use_sparsity=*/use_sparsity);
  }

  uint32_t numNonzerosInOutputImpl() const final { return (*_outputs)[0].len; }

  void forwardImpl(uint32_t vec_index, const BoltVector* labels) final {
    _layer->forward(_predecessor->getOutputVector(vec_index),
                    this->getOutputVectorImpl(vec_index), labels);
  }

  void backpropagateImpl(uint32_t vec_index) final {
    // We are checking whether predecessor has gradients or not rather than its
    // an input ot not because,this way will be helpful to calculate gradients
    // for input in getInputGradientsSingle.
    if (!_predecessor->getOutputVector(vec_index).gradients) {
      _layer->backpropagateInputLayer(_predecessor->getOutputVector(vec_index),
                                      this->getOutputVectorImpl(vec_index));
    } else {
      _layer->backpropagate(_predecessor->getOutputVector(vec_index),
                            this->getOutputVectorImpl(vec_index));
    }
  }

  void updateParametersImpl(float learning_rate, uint32_t batch_cnt) final {
    // TODO(Nicholas): Abstract away these constants
    _layer->updateParameters(learning_rate, batch_cnt, BETA1, BETA2, EPS);
  }

  BoltVector& getOutputVectorImpl(uint32_t vec_index) final {
    return (*_outputs)[vec_index];
  }

  void cleanupAfterBatchProcessingImpl() final { _outputs = std::nullopt; }

  std::vector<NodePtr> getPredecessorsImpl() const final {
    return {_predecessor};
  }

  void summarizeImpl(std::stringstream& summary, bool detailed) const final {
    summary << _predecessor->name() << " -> " << name()
            << " (FullyConnected): ";
    _layer->buildLayerSummary(summary, detailed);
  }

  std::string type() const final { return "fc"; }

  NodeState getState() const final {
    if (_predecessor == nullptr && _layer == nullptr && !_outputs.has_value()) {
      return NodeState::Constructed;
    }
    if (_predecessor != nullptr && _layer == nullptr && !_outputs.has_value()) {
      return NodeState::PredecessorsSet;
    }
    if (_predecessor != nullptr && _layer != nullptr && !_outputs.has_value()) {
      return NodeState::Compiled;
    }
    if (_predecessor != nullptr && _layer != nullptr && _outputs.has_value()) {
      return NodeState::PreparedForBatchProcessing;
    }
    throw exceptions::NodeStateMachineError(
        "FullyConnectedNode is in an invalid internal state");
  }

  // Private constructor for cereal. Must create dummy config since no default
  // constructor exists for layer config.
  FullyConnectedNode() : _config(std::nullopt) {}

  friend class cereal::access;
  template <class Archive>
  void serialize(Archive& archive) {
    archive(cereal::base_class<Node>(this), _layer, _config, _predecessor);
  }
  // One of _layer and _config will always be nullptr/nullopt while the
  // other will contain data
  std::shared_ptr<FullyConnectedLayer> _layer;
  std::optional<FullyConnectedLayerConfig> _config;
  std::optional<BoltBatch> _outputs;

  NodePtr _predecessor;
};

using FullyConnectedNodePtr = std::shared_ptr<FullyConnectedNode>;

}  // namespace thirdai::bolt

CEREAL_REGISTER_TYPE(thirdai::bolt::FullyConnectedNode)