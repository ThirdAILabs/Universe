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

namespace thirdai::automl::deployment {

/**
 * This config provides the structure for instantiating the bolt dag.
 * Args:
 *    - input_names: the names of the inputs to the dag. The input nodes
 *      themselves are constructed by the dataset factory, this is just used to
 *      reference them. The ith input node created by the dataset factory
 *      corresponds to the ith input here. The number of input names must match
 *      the number of input nodes from the dataset factory.
 *    - nodes: the nodes of the dag. Note that the last node must be the output
 *      of the DAG.
 *    - loss: The loss function to use.
 */
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
      const UserInputMap& user_specified_parameters) const {
    if (_input_names.size() != inputs.size()) {
      throw std::invalid_argument(
          "Number of inputs in model config does not match number of inputs "
          "returned from data loader.");
    }

    PredecessorsMap predecessors;
    for (uint32_t i = 0; i < _input_names.size(); i++) {
      predecessors.insert(/* name= */ _input_names[i], /* node= */ inputs[i]);
    }

    for (uint32_t i = 0; i < _nodes.size() - 1; i++) {
      auto node =
          _nodes[i]->createNode(predecessors, user_specified_parameters);
      predecessors.insert(/* name= */ _nodes[i]->name(), /* node= */ node);
    }

    auto output =
        _nodes.back()->createNode(predecessors, user_specified_parameters);
    // This is to check that there is not another node with this name.
    predecessors.insert(/* name= */ _nodes.back()->name(), /* node= */ output);

    auto model = std::make_shared<bolt::BoltGraph>(inputs, output);

    model->compile(_loss);

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

}  // namespace thirdai::automl::deployment