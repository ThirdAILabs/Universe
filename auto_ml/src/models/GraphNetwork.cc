#include "GraphNetwork.h"
#include <auto_ml/src/dataset_factories/udt/GraphDatasetFactory.h>
#include <limits>
#include <bolt/src/graph/nodes/Concatenate.h>
#include <bolt/src/graph/nodes/Embedding.h>
#include <bolt/src/graph/nodes/FullyConnected.h>

namespace thirdai::automl::models {

bolt::BoltGraphPtr createGNN(uint32_t input_dim, uint32_t output_dim,
    uint32_t max_neighbors) {

  auto node_features_input = bolt::Input::make(input_dim);

  auto neighbor_token_input = bolt::Input::makeTokenInput(
      /*expected_dim = */ std::numeric_limits<uint64_t>::max(),
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

  concat_node->setConcatenatedNodes(/* nodes = */{hidden_1, embedding_1});

  auto hidden_3 = bolt::FullyConnectedNode::make(
      /*dim = */ 256, /* sparsity = */ 0.5, /* activation = */ "relu", /* sampling_config = */ std::make_shared<bolt::RandomSamplingConfig>());

  hidden_3->addPredecessor(concat_node);

  // auto hidden_4 = bolt::FullyConnectedNode::makeAutotuned(
  //     /*dim=*/256, /*sparsity=*/1, /*activation=*/"relu");

  // hidden_4->addPredecessor(hidden_3);

  auto output = bolt::FullyConnectedNode::makeAutotuned(
      /* dim = */ output_dim, /* sparsity = */ 1, /* activation =*/ "softmax");

  output->addPredecessor(hidden_3);

  auto graph = std::make_shared<bolt::BoltGraph>(
      /* inputs= */ {node_features_input, neighbor_token_input}, output);

  graph->compile(
      bolt::CategoricalCrossEntropyLoss::makeCategoricalCrossEntropyLoss());

  return graph;
}

GraphNetwork GraphNetwork::create(
    data::ColumnDataTypes data_types, std::string target_col,
               std::optional<uint32_t> n_target_classes, bool integer_target,
               char delimiter, uint32_t max_neighbors) {

  auto graph_dataset_factory =
      std::make_shared<data::GraphDatasetFactory>(data_types, target_col, n_target_classes, integer_target, delimiter);

  bolt::BoltGraphPtr model = createGNN(
      /* input_dim = */ graph_dataset_factory->getInputDims().at(0),
      /* output_dim = */ graph_dataset_factory->getLabelDim(), /* max_neighbors = */ max_neighbors);

  TrainEvalParameters train_eval_parameters(
      /* rebuild_hash_tables_interval= */ std::nullopt,
      /* reconstruct_hash_functions_interval= */ std::nullopt,
      /* default_batch_size= */ DEFAULT_INFERENCE_BATCH_SIZE,
      /* freeze_hash_tables= */ true,
      /* prediction_threshold= */ std::nullopt);

  return GraphNetwork({graph_dataset_factory, model,
                   getOutputProcessor(dataset_config), train_eval_parameters});
}

OutputProcessorPtr GraphNetwork::getOutputProcessor(
    const data::GraphConfigPtr& dataset_config) {
  if (dataset_config->_n_target_classes == 2) {
    return BinaryOutputProcessor::make();
  }

  return CategoricalOutputProcessor::make();
}

}  // namespace thirdai::automl::models