#pragma once

#include <bolt/src/graph/Graph.h>
#include <bolt/src/graph/Node.h>
#include <bolt/src/graph/nodes/FullyConnected.h>
#include <bolt/src/layers/FullyConnectedLayer.h>
#include <bolt/src/layers/SamplingConfig.h>
#include <bolt/src/loss_functions/LossFunctions.h>
#include <bolt/src/networks/FullyConnectedNetwork.h>
#include <memory>
#include <optional>
#include <stdexcept>
namespace thirdai::bolt {

class CommonNetworks {
 public:
  static BoltGraphPtr FullyConnected(
      uint32_t input_dim, std::vector<FullyConnectedNodePtr> layers) {
    if (layers.empty()) {
      throw std::invalid_argument(
          "CommonNetworks::FullyConnected: Must pass at least one layer.");
    }

    auto input_layer = std::make_shared<Input>(input_dim);
    NodePtr prev_layer = input_layer;

    for (auto& layer : layers) {
      layer->addPredecessor(prev_layer);
      prev_layer = layer;
    }

    auto model = std::make_shared<BoltGraph>(std::vector<InputPtr>{input_layer},
                                             layers.back());

    return model;
  }
};

}  // namespace thirdai::bolt