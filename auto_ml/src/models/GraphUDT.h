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
      uint32_t num_neighbours,
      std::optional<std::vector<std::string>> relationship_columns =
          std::nullopt,
      bool integer_target = false, bool numerical_context = false,
      bool features_context = false, uint32_t k_hop = 1, char delimeter = ',',
      std::optional<std::unordered_map<std::string, std::vector<std::string>>>
          adj_list = std::nullopt);

  static OutputProcessorPtr getOutputProcessor(
      const data::GraphConfigPtr& dataset_config);

 private:
  explicit GraphUDT(ModelPipeline&& model) : ModelPipeline(model) {}

  static bolt::BoltGraphPtr buildGraphUDTBoltGraph(
      const std::vector<uint32_t>& input_dims, uint32_t output_dim,
      uint32_t num_neighbours);
};

}  // namespace thirdai::automl::models