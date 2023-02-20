#include "GraphUDT.h"

namespace thirdai::automl::models {

GraphUDT GraphUDT::buildGraphUDT(
    data::ColumnDataTypes data_types, std::string graph_file_name,
    std::string source, std::string target, uint32_t n_target_classes,
    uint32_t max_neighbours,
    std::optional<std::vector<std::string>> relationship_columns,
    bool integer_target, bool numerical_context, bool features_context,
    uint32_t k_hop, char delimeter,
    std::optional<std::unordered_map<std::string, std::vector<std::string>>>
        adj_list) {
  auto dataset_config = std::make_shared<data::GraphConfig>(
      std::move(data_types), std::move(graph_file_name), std::move(source),
      std::move(target), n_target_classes, max_neighbours,
      std::move(relationship_columns), integer_target, numerical_context,
      features_context, k_hop, delimeter, std::move(adj_list));

  auto graph_dataset_factory =
      std::make_shared<data::GraphDatasetFactory>(dataset_config);

  bolt::BoltGraphPtr model;
  model = GraphUDT::buildGraphUDTBoltGraph(
      graph_dataset_factory->getInputDims(),
      graph_dataset_factory->getLabelDim(), max_neighbours);

  TrainEvalParameters train_eval_parameters(
      /* rebuild_hash_tables_interval= */ std::nullopt,
      /* reconstruct_hash_functions_interval= */ std::nullopt,
      /* default_batch_size= */ DEFAULT_INFERENCE_BATCH_SIZE,
      /* freeze_hash_tables= */ true,
      /* prediction_threshold= */ std::nullopt);

  return GraphUDT({graph_dataset_factory, model,
                   getOutputProcessor(dataset_config), train_eval_parameters});
}

OutputProcessorPtr GraphUDT::getOutputProcessor(
    const data::GraphConfigPtr& dataset_config) {
  if (dataset_config->_n_target_classes == 2) {
    return BinaryOutputProcessor::make();
  }

  return CategoricalOutputProcessor::make();
}

// TODO(YASH): Autotune the dimension and sparsity.
bolt::BoltGraphPtr GraphUDT::buildGraphUDTBoltGraph(
    const std::vector<uint32_t>& input_dims, uint32_t output_dim,
    uint32_t max_neighbours) {
  std::vector<bolt::InputPtr> input_nodes;
  input_nodes.reserve(input_dims.size());
  for (uint32_t input_dim : input_dims) {
    input_nodes.push_back(bolt::Input::make(input_dim));
  }
  auto token_input = bolt::Input::makeTokenInput(
      /*expected_dim=*/4294967295,
      /*num_tokens_range*/ {max_neighbours, max_neighbours});

  input_nodes.push_back(token_input);

  auto embedding_1 = bolt::EmbeddingNode::make(
      /*num_embedding_lookups=*/4, /*lookup_size=*/64,
      /*log_embedding_block_size=*/29, /*reduction=*/"concatenation",
      /*num_tokens_per_input=*/max_neighbours);

  auto hidden_1 = bolt::FullyConnectedNode::makeAutotuned(
      /*dim=*/512, /*sparsity=*/1, /*activation=*/"relu");

  hidden_1->addPredecessor(input_nodes[0]);

  embedding_1->addInput(token_input);

  auto hidden_2 = bolt::FullyConnectedNode::makeAutotuned(
      /*dim=*/256, /*sparsity=*/1, /*activation=*/"relu");

  hidden_2->addPredecessor(hidden_1);

  auto concat_node = bolt::ConcatenateNode::make();

  concat_node->setConcatenatedNodes(/*nodes=*/{hidden_2, embedding_1});

  auto hidden_3 = bolt::FullyConnectedNode::makeAutotuned(
      /*dim=*/256, /*sparsity=*/1, /*activation=*/"relu");

  hidden_3->addPredecessor(concat_node);

  auto hidden_4 = bolt::FullyConnectedNode::makeAutotuned(
      /*dim=*/256, /*sparsity=*/1, /*activation=*/"relu");

  hidden_4->addPredecessor(hidden_3);

  auto output = bolt::FullyConnectedNode::makeAutotuned(
      /*dim=*/output_dim, /*sparsity=*/1, /*activation=*/"softmax");

  output->addPredecessor(hidden_4);

  auto graph = std::make_shared<bolt::BoltGraph>(
      /* inputs= */ input_nodes, output);

  graph->compile(
      bolt::CategoricalCrossEntropyLoss::makeCategoricalCrossEntropyLoss());

  return graph;
}

}  // namespace thirdai::automl::models