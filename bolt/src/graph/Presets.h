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

/**
 * Shortcuts for commonly used models and layers.
 */
class Presets {
 public:
  static FullyConnectedNodePtr FullyConnectedLayer(
      uint32_t dim, float sparsity, std::string activation_function) {
    return std::make_shared<FullyConnectedNode>(dim, sparsity,
                                                activation_function);
  }

  static FullyConnectedNodePtr FullyConnectedLayer(
      uint32_t dim, float sparsity, std::string activation_function,
      uint32_t num_tables, uint32_t hashes_per_table, uint32_t reservoir_size) {
    auto sampling_config = std::make_shared<DWTASamplingConfig>(
        num_tables, hashes_per_table, reservoir_size);
    return std::make_shared<FullyConnectedNode>(
        dim, sparsity, activation_function, sampling_config);
  }

  static BoltGraphPtr FullyConnectedNetwork(
      uint32_t input_dim, std::vector<FullyConnectedNodePtr> layers,
      std::shared_ptr<LossFunction> loss) {
    if (layers.empty()) {
      throw std::invalid_argument(
          "Presets::FullyConnectedNetwork: Must pass at least one layer.");
    }

    auto input_layer = std::make_shared<Input>(input_dim);
    NodePtr prev_layer = input_layer;

    for (auto& layer : layers) {
      layer->addPredecessor(prev_layer);
      prev_layer = layer;
    }

    auto model = std::make_shared<BoltGraph>(std::vector<InputPtr>{input_layer},
                                             layers.back());

    model->compile(std::move(loss));

    return model;
  }
};

}  // namespace thirdai::bolt