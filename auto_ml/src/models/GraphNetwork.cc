#include "GraphNetwork.h"
#include "Utils.h"
#include <bolt/src/graph/DatasetContext.h>
#include <bolt/src/graph/nodes/Concatenate.h>
#include <bolt/src/graph/nodes/Embedding.h>
#include <bolt/src/graph/nodes/FullyConnected.h>
#include <auto_ml/src/dataset_factories/udt/GraphDatasetFactory.h>
#include <auto_ml/src/models/OutputProcessor.h>
#include <limits>
#include <stdexcept>

namespace thirdai::automl::models {

bolt::BoltGraphPtr createGNN(std::vector<uint32_t> input_dims,
                             uint32_t output_dim, uint32_t max_neighbors) {
  auto node_features_input = bolt::Input::make(input_dims.at(0));

  auto neighbor_token_input = bolt::Input::makeTokenInput(
      /*expected_dim = */ input_dims.at(1),
      /*num_tokens_range = */ {0, max_neighbors});

  auto embedding_1 = bolt::EmbeddingNode::make(
      /*num_embedding_lookups = */ 4, /* lookup_size = */ 64,
      /*log_embedding_block_size = */ 20, /*reduction = */ "concatenation",
      /*num_tokens_per_input = */ max_neighbors);

  auto hidden_1 = bolt::FullyConnectedNode::makeAutotuned(
      /*dim = */ 512, /*sparsity = */ 1, /*activation = */ "relu");

  hidden_1->addPredecessor(node_features_input);

  embedding_1->addInput(neighbor_token_input);

  // auto hidden_2 = bolt::FullyConnectedNode::makeAutotuned(
  //     /*dim=*/256, /*sparsity=*/1, /*activation=*/"relu");

  // hidden_2->addPredecessor(hidden_1);

  auto concat_node = bolt::ConcatenateNode::make();

  concat_node->setConcatenatedNodes(/* nodes = */ {hidden_1, embedding_1});

  auto hidden_3 = bolt::FullyConnectedNode::make(
      /*dim = */ 256, /* sparsity = */ 0.5, /* activation = */ "relu",
      /* sampling_config = */ std::make_shared<bolt::RandomSamplingConfig>());

  hidden_3->addPredecessor(concat_node);

  // auto hidden_4 = bolt::FullyConnectedNode::makeAutotuned(
  //     /*dim=*/256, /*sparsity=*/1, /*activation=*/"relu");

  // hidden_4->addPredecessor(hidden_3);

  auto output = bolt::FullyConnectedNode::makeAutotuned(
      /* dim = */ output_dim, /* sparsity = */ 1, /* activation =*/"softmax");

  output->addPredecessor(hidden_3);

  std::vector<bolt::InputPtr> inputs = {node_features_input,
                                        neighbor_token_input};

  auto graph = std::make_shared<bolt::BoltGraph>(inputs, output);

  graph->compile(
      bolt::CategoricalCrossEntropyLoss::makeCategoricalCrossEntropyLoss());

  return graph;
}

GraphNetwork GraphNetwork::create(data::ColumnDataTypes data_types,
                                  std::string target_col,
                                  uint32_t n_target_classes,
                                  bool integer_target, char delimiter,
                                  uint32_t max_neighbors, uint32_t k_hop,
                                  bool store_node_features) {
  verifyDataTypesContainTarget(data_types, target_col);

  auto [output_processor, regression_binning] =
      getOutputProcessor(data_types, target_col, n_target_classes);
  if (regression_binning.has_value()) {
    throw exceptions::NotImplemented(
        "We do not yet support regression on graphs.");
  }

  if (!integer_target) {
    throw exceptions::NotImplemented(
        "We do not yet support non integer classes on graphs.");
  }

  auto graph_dataset_factory = std::make_shared<data::GraphDatasetFactory>(
      data_types, target_col, n_target_classes, delimiter, max_neighbors, k_hop,
      store_node_features);

  bolt::BoltGraphPtr model = createGNN(
      /* input_dims = */ graph_dataset_factory->getInputDims(),
      /* output_dim = */ graph_dataset_factory->getLabelDim(),
      /* max_neighbors = */ max_neighbors);

  TrainEvalParameters train_eval_parameters =
      defaultTrainEvalParams(/* freeze_hash_tables = */ false);

  return GraphNetwork(graph_dataset_factory, model, output_processor,
                      train_eval_parameters);
}

}  // namespace thirdai::automl::models