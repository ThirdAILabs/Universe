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
      const std::vector<uint32_t>& input_dims,
      const UserInputMap& user_specified_parameters) const;

  void save(const std::string& filename);

  static std::shared_ptr<ModelConfig> load(const std::string& filename);

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
