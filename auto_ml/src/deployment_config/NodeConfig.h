#pragma once

#include <cereal/access.hpp>
#include <cereal/types/base_class.hpp>
#include <cereal/types/memory.hpp>
#include <cereal/types/optional.hpp>
#include <cereal/types/polymorphic.hpp>
#include "HyperParameter.h"
#include <bolt/src/graph/Node.h>
#include <bolt/src/graph/nodes/FullyConnected.h>
#include <bolt/src/layers/SamplingConfig.h>
#include <optional>
#include <stdexcept>

namespace thirdai::automl::deployment_config {

class PredecessorsMap {
 public:
  void update(const std::string& name, bolt::NodePtr node) {
    if (_discovered_nodes.count(name)) {
      throw std::invalid_argument("Cannot have multiple nodes with the name '" +
                                  name + "' in the model config.");
    }
    _discovered_nodes[name] = std::move(node);
  }

  bolt::NodePtr getNode(const std::string& name) const {
    if (!_discovered_nodes.count(name)) {
      throw std::invalid_argument("Cannot find node with name '" + name +
                                  "' in already discovered nodes.");
    }
    return _discovered_nodes.at(name);
  }

 private:
  std::unordered_map<std::string, bolt::NodePtr> _discovered_nodes;
};

class NodeConfig {
 public:
  explicit NodeConfig(std::string name) : _name(std::move(name)) {}

  const std::string& name() const { return _name; }

  virtual bolt::NodePtr createNode(
      const PredecessorsMap& possible_predecessors,
      const std::optional<std::string>& option,
      const UserInputMap& user_specified_parameters) const = 0;

  virtual ~NodeConfig() = default;

 protected:
  // Private constructor for cereal.
  NodeConfig() {}

 private:
  std::string _name;

  friend class cereal::access;
  template <class Archive>
  void serialize(Archive& archive) {
    archive(_name);
  }
};

using NodeConfigPtr = std::shared_ptr<NodeConfig>;

class FullyConnectedNodeConfig final : public NodeConfig {
 public:
  FullyConnectedNodeConfig(
      std::string name, HyperParameterPtr<uint32_t> dim,
      HyperParameterPtr<float> sparsity,
      HyperParameterPtr<std::string> activation, std::string predecessor_name,
      std::optional<HyperParameterPtr<bolt::SamplingConfigPtr>> sampling_config)
      : NodeConfig(std::move(name)),
        _dim(std::move(dim)),
        _sparsity(std::move(sparsity)),
        _activation(std::move(activation)),
        _sampling_config(std::move(sampling_config)),
        _predecessor_name(std::move(predecessor_name)) {}

  FullyConnectedNodeConfig(std::string name, HyperParameterPtr<uint32_t> dim,
                           HyperParameterPtr<std::string> activation,
                           std::string predecessor_name)
      : NodeConfig(std::move(name)),
        _dim(std::move(dim)),
        _sparsity(ConstantParameter<float>::make(1.0)),
        _activation(std::move(activation)),
        _sampling_config(std::nullopt),
        _predecessor_name(std::move(predecessor_name)) {}

  bolt::NodePtr createNode(
      const PredecessorsMap& possible_predecessors,
      const std::optional<std::string>& option,
      const UserInputMap& user_specified_parameters) const final {
    uint32_t dim = _dim->resolve(option, user_specified_parameters);
    float sparsity = _sparsity->resolve(option, user_specified_parameters);
    std::string activation =
        _activation->resolve(option, user_specified_parameters);

    bolt::FullyConnectedNodePtr node;
    if (_sampling_config) {
      bolt::SamplingConfigPtr sampling_config =
          (*_sampling_config)->resolve(option, user_specified_parameters);

      node = bolt::FullyConnectedNode::make(dim, sparsity, activation,
                                            sampling_config);
    } else {
      node = bolt::FullyConnectedNode::makeAutotuned(dim, sparsity, activation);
    }

    node->addPredecessor(possible_predecessors.getNode(_predecessor_name));

    return node;
  }

 private:
  HyperParameterPtr<uint32_t> _dim;
  HyperParameterPtr<float> _sparsity;
  HyperParameterPtr<std::string> _activation;
  std::optional<HyperParameterPtr<bolt::SamplingConfigPtr>> _sampling_config;
  std::string _predecessor_name;

  // Private constructor for cereal.
  FullyConnectedNodeConfig() {}

  friend class cereal::access;
  template <class Archive>
  void serialize(Archive& archive) {
    archive(cereal::base_class<NodeConfig>(this), _dim, _sparsity, _activation,
            _sampling_config, _predecessor_name);
  }
};

}  // namespace thirdai::automl::deployment_config

CEREAL_REGISTER_TYPE(
    thirdai::automl::deployment_config::FullyConnectedNodeConfig)