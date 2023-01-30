#pragma once

#include <bolt/src/callbacks/Callback.h>
#include <bolt/src/graph/DatasetContext.h>
#include <bolt/src/graph/ExecutionConfig.h>
#include <bolt/src/graph/Graph.h>
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
      std::string source, std::string target,
      std::vector<std::string> relationship_columns, uint32_t n_target_classes,
      bool neighbourhood_context = false, bool label_context = false,
      uint32_t kth_neighbourhood = 0, char delimeter = ',') {
    auto dataset_config = std::make_shared<data::GraphConfig>(
        std::move(data_types), std::move(graph_file_name), std::move(source),
        std::move(target), std::move(relationship_columns), n_target_classes,
        neighbourhood_context, label_context, kth_neighbourhood, delimeter);

    auto graph_dataset_factory =
        std::make_shared<data::GraphDatasetFactory>(dataset_config);

    graph_dataset_factory->prepareTheBatchProcessor();

    bolt::BoltGraphPtr model;
    model = UniversalDeepTransformer::buildUDTBoltGraph(
        /* input_dims= */ graph_dataset_factory->getInputDims(),
        /* output_dim= */ graph_dataset_factory->getLabelDim(),
        /* hidden_layer_dim= */ 1024);

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
  explicit GraphUDT(ModelPipeline&& model) : ModelPipeline(model) {}
};

}  // namespace thirdai::automl::models