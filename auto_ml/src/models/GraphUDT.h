#pragma once

#include <bolt/src/callbacks/Callback.h>
#include <bolt/src/graph/DatasetContext.h>
#include <bolt/src/graph/ExecutionConfig.h>
#include <bolt/src/graph/Graph.h>
#include <bolt/src/graph/nodes/Concatenate.h>
#include <bolt/src/graph/nodes/Embedding.h>
#include <bolt/src/graph/nodes/FullyConnected.h>
#include <bolt/src/graph/nodes/Input.h>
#include <auto_ml/src/dataset_factories/udt/GraphConfig.h>
#include <auto_ml/src/dataset_factories/udt/GraphDatasetFactory.h>
#include <auto_ml/src/models/ModelPipeline.h>
#include <auto_ml/src/models/UniversalDeepTransformer.h>
#include <dataset/src/DataSource.h>
#include <dataset/src/dataset_loaders/DatasetLoader.h>
#include <memory>
#include <optional>
#include <utility>
namespace thirdai::automl::models {

class GraphUDT : public ModelPipeline {
  static constexpr const uint32_t DEFAULT_INFERENCE_BATCH_SIZE = 2048;

 public:
  static GraphUDT buildGraphUDT(
      data::ColumnDataTypes data_types, std::string graph_file_name,
      std::string source, std::string target, uint32_t n_target_classes,
      uint32_t max_neighbours,
      std::optional<std::vector<std::string>> relationship_columns =
          std::nullopt,
      bool neighbourhood_context = false, bool label_context = false,
      uint32_t kth_neighbourhood = 0, char delimeter = ',',
      std::optional<std::unordered_map<std::string, std::vector<std::string>>>
          adj_list = std::nullopt) {
    auto dataset_config = std::make_shared<data::GraphConfig>(
        std::move(data_types), std::move(graph_file_name), std::move(source),
        std::move(target), n_target_classes, max_neighbours,
        std::move(relationship_columns), neighbourhood_context, label_context,
        kth_neighbourhood, delimeter, std::move(adj_list));

    auto graph_dataset_factory =
        std::make_shared<data::GraphDatasetFactory>(dataset_config);

    graph_dataset_factory->prepareTheBatchProcessor();

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
                     getOutputProcessor(dataset_config),
                     train_eval_parameters});
  }
  static OutputProcessorPtr getOutputProcessor(
      const data::GraphConfigPtr& dataset_config) {
    if (dataset_config->_n_target_classes == 2) {
      return BinaryOutputProcessor::make();
    }

    return CategoricalOutputProcessor::make();
  }

 private:
  static bolt::BoltGraphPtr buildGraphUDTBoltGraph(
      const std::vector<uint32_t>& input_dims, uint32_t output_dim,
      uint32_t max_neighbours) {
    std::vector<bolt::InputPtr> input_nodes;
    input_nodes.reserve(input_dims.size());
    for (uint32_t input_dim : input_dims) {
      input_nodes.push_back(bolt::Input::make(input_dim));
    }
    auto token_input = bolt::Input::makeTokenInput(
        4294967295, {max_neighbours, max_neighbours});

    input_nodes.push_back(token_input);

    auto embedding_1 = bolt::EmbeddingNode::make(
        4, 64, 4294967295, "concatenation", max_neighbours);

    auto hidden_1 = bolt::FullyConnectedNode::makeAutotuned(512, 0.2, "relu");

    hidden_1->addPredecessor(input_nodes[0]);

    embedding_1->addInput(token_input);

    auto hidden_2 = bolt::FullyConnectedNode::makeAutotuned(256, 0.2, "relu");

    hidden_2->addPredecessor(hidden_1);

    auto concat_node = bolt::ConcatenateNode::make();

    concat_node->setConcatenatedNodes({hidden_2, embedding_1});

    auto hidden_3 = bolt::FullyConnectedNode::makeAutotuned(256, 0.2, "relu");

    hidden_3->addPredecessor(concat_node);

    auto hidden_4 = bolt::FullyConnectedNode::makeAutotuned(256, 0.2, "relu");

    hidden_4->addPredecessor(hidden_3);

    auto output =
        bolt::FullyConnectedNode::makeAutotuned(output_dim, 1, "softmax");

    output->addPredecessor(hidden_4);

    auto graph = std::make_shared<bolt::BoltGraph>(
        /* inputs= */ input_nodes, output);

    graph->compile(
        bolt::CategoricalCrossEntropyLoss::makeCategoricalCrossEntropyLoss());

    return graph;
  }
  explicit GraphUDT(ModelPipeline&& model) : ModelPipeline(model) {}
};

}  // namespace thirdai::automl::models