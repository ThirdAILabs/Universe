#pragma once

#include "NodeConfig.h"
#include <bolt/src/graph/Graph.h>
#include <bolt/src/graph/Node.h>
#include <bolt/src/graph/nodes/Input.h>
#include <bolt/src/loss_functions/LossFunctions.h>
#include <memory>
#include <stdexcept>
#include <vector>

namespace thirdai::automl::deployment_config {

class ModelConfig {
 public:
  ModelConfig(std::vector<std::string> input_names,
              std::vector<NodeConfigPtr> nodes,
              HyperParameterPtr<std::shared_ptr<bolt::LossFunction>> loss)
      : _input_names(std::move(input_names)),
        _nodes(std::move(nodes)),
        _loss(std::move(loss)) {}

  bolt::BoltGraphPtr createModel(
      std::vector<bolt::InputPtr> inputs, const std::string& option,
      const std::unordered_map<std::string, UserParameterInput>&
          user_specified_parameters) const {
    if (_input_names.size() != inputs.size()) {
      throw std::invalid_argument(
          "Expected number of inputs does not match number of inputs returned "
          "from data loader.");
    }

    PredecessorsMap predecessors;
    for (uint32_t i = 0; i < _input_names.size(); i++) {
      predecessors.update(_input_names[i], inputs[i]);
    }

    for (uint32_t i = 0; i < _nodes.size() - 1; i++) {
      auto node = _nodes[i]->createNode(predecessors, option,
                                        user_specified_parameters);
      predecessors.update(_nodes[i]->name(), node);
    }

    auto output = _nodes.back()->createNode(predecessors, option,
                                            user_specified_parameters);

    auto model = std::make_shared<bolt::BoltGraph>(inputs, output);

    auto loss = _loss->resolve(option, user_specified_parameters);
    model->compile(loss);

    return model;
  }

 private:
  std::vector<std::string> _input_names;
  std::vector<NodeConfigPtr> _nodes;
  HyperParameterPtr<std::shared_ptr<bolt::LossFunction>> _loss;
};

}  // namespace thirdai::automl::deployment_config