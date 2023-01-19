#pragma once

#include <bolt/src/callbacks/Callback.h>
#include <auto_ml/src/dataset_factories/udt/GraphConfig.h>
#include <auto_ml/src/dataset_factories/udt/GraphDatasetFactory.h>
#include <auto_ml/src/models/ModelPipeline.h>
#include <memory>
#include <utility>
namespace thirdai::automl::models {

class GraphUDT {
 public:
  explicit GraphUDT(bolt::BoltGraphPtr model) : _model(std::move(model)) {}

  // static GraphUDT buildGraphUDT(
  //     const data::ColumnDataTypes& data_types,
  //     const std::string& graph_file_name, const std::string& source,
  //     const std::string& target,
  //     const std::vector<std::string>& relationship_columns,
  //     uint32_t n_target_classes, bool neighbourhood_context = false,
  //     bool label_context = false, uint32_t kth_neighbourhood = 0,
  //     char delimeter = ',') {
  //   auto dataset_config = std::make_shared<data::GraphConfig>(
  //       data_types, graph_file_name, source, target, relationship_columns,
  //       n_target_classes, neighbourhood_context, label_context,
  //       kth_neighbourhood, delimeter);

  //   auto graph_dataset_factory =
  //       std::make_shared<data::GraphDatasetFactory>(dataset_config);
  // }

 private:
  bolt::BoltGraphPtr _model;
};

}  // namespace thirdai::automl::models