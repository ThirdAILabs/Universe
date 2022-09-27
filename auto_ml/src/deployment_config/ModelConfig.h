#pragma once

#include <cereal/access.hpp>
#include <cereal/types/vector.hpp>
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
              std::shared_ptr<bolt::LossFunction> loss)
      : _input_names(std::move(input_names)),
        _nodes(std::move(nodes)),
        _loss(std::move(loss)) {}

  bolt::BoltGraphPtr createModel(
      std::vector<bolt::InputPtr> inputs,
      const std::optional<std::string>& option,
      const UserInputMap& user_specified_parameters) const {
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

// Disable the model summary in the release, but print it out for internal use.
#if THIRDAI_EXPOSE_ALL
    model->compile(_loss);
#else
    model->compile(_loss, /* print_when_done= */ false);
#endif

    return model;
  }

 private:
  std::vector<std::string> _input_names;
  std::vector<NodeConfigPtr> _nodes;
  std::shared_ptr<bolt::LossFunction> _loss;

  // Private constructor for cereal.
  ModelConfig() {}

  friend class cereal::access;
  template <class Archive>
  void serialize(Archive& archive) {
    archive(_input_names, _nodes, _loss);
  }
};

using ModelConfigPtr = std::shared_ptr<ModelConfig>;

}  // namespace thirdai::automl::deployment_config