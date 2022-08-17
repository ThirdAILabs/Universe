#include <bolt/src/graph/Graph.h>
#include <bolt/src/graph/nodes/FullyConnected.h>
#include <bolt/src/loss_functions/LossFunctions.h>
#include <bolt/src/networks/FullyConnectedNetwork.h>
#include <memory>
namespace thirdai::bolt {

class FullyConnectedGraphNetwork {
 public:
  static BoltGraphPtr build(
      uint32_t input_dim,
      const std::vector<std::pair<uint32_t, float>>& hidden_dims_and_sparsities,
      uint32_t output_dim, float output_sparsity,
      std::string output_activation = "softmax",
      std::shared_ptr<LossFunction> loss =
          std::make_shared<CategoricalCrossEntropyLoss>()) {
    auto input_layer = std::make_shared<Input>(input_dim);
    NodePtr prev_layer = input_layer;

    for (auto [dim, sparsity] : hidden_dims_and_sparsities) {
      auto hidden_layer = std::make_shared<FullyConnectedNode>(
          /* dim= */ dim,
          /* sparsity= */ sparsity,
          /* activation= */ "relu");

      hidden_layer->addPredecessor(prev_layer);
      prev_layer = hidden_layer;
    }

    auto output_layer = std::make_shared<FullyConnectedNode>(
        /* dim= */ output_dim, /* sparsity= */ output_sparsity,
        /* activation= */ output_activation);

    output_layer->addPredecessor(prev_layer);

    auto model = std::make_shared<BoltGraph>(std::vector<InputPtr>{input_layer},
                                             output_layer);

    model->compile(std::move(loss));

    return model;
  }
};

}  // namespace thirdai::bolt