#pragma once

#include <auto_ml/src/dataset_factories/udt/DataTypes.h>
#include <auto_ml/src/models/ModelPipeline.h>
#include <memory>

namespace thirdai::automl::models {

class GraphNetwork : public ModelPipeline {
 public:
  static GraphNetwork create(data::ColumnDataTypes data_types,
                             std::string target_col, uint32_t n_target_classes,
                             bool integer_target, char delimiter,
                             uint32_t max_neighbors, bool store_node_features);

 private:
  // Inherit ModelPipeline constructor privately
  using ModelPipeline::ModelPipeline;
};

using GraphNetworkPtr = std::shared_ptr<GraphNetwork>;

}  // namespace thirdai::automl::models