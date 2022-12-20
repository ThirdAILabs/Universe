#include "NodeConfig.h"
#include <cereal/archives/portable_binary.hpp>

namespace thirdai::automl::deployment {

FullyConnectedNodeConfig::FullyConnectedNodeConfig(
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

FullyConnectedNodeConfig::FullyConnectedNodeConfig(
    std::string name, HyperParameterPtr<uint32_t> dim,
    HyperParameterPtr<std::string> activation, std::string predecessor_name)
    : NodeConfig(std::move(name)),
      _dim(std::move(dim)),
      _sparsity(ConstantParameter<float>::make(1.0)),
      _activation(std::move(activation)),
      _sampling_config(std::nullopt),
      _predecessor_name(std::move(predecessor_name)) {}

bolt::NodePtr FullyConnectedNodeConfig::createNode(
    const PredecessorsMap& possible_predecessors,
    const UserInputMap& user_specified_parameters) const {
  uint32_t dim = _dim->resolve(user_specified_parameters);
  float sparsity = _sparsity->resolve(user_specified_parameters);
  std::string activation = _activation->resolve(user_specified_parameters);

  bolt::FullyConnectedNodePtr node;
  if (_sampling_config) {
    bolt::SamplingConfigPtr sampling_config =
        (*_sampling_config)->resolve(user_specified_parameters);

    node = bolt::FullyConnectedNode::make(dim, sparsity, activation,
                                          sampling_config);
  } else {
    node = bolt::FullyConnectedNode::makeAutotuned(dim, sparsity, activation);
  }

  node->addPredecessor(possible_predecessors.get(_predecessor_name));

  return node;
}

}  // namespace thirdai::automl::deployment

CEREAL_REGISTER_TYPE(thirdai::automl::deployment::FullyConnectedNodeConfig)
