#pragma once

#include <auto_ml/src/dataset_factories/udt/DataTypes.h>
#include <auto_ml/src/dataset_factories/udt/GraphDatasetFactory.h>
#include <auto_ml/src/models/ModelPipeline.h>
#include <dataset/src/DataSource.h>
#include <memory>

namespace thirdai::automl::models {

class GraphNetwork : public ModelPipeline {
 public:
  static GraphNetwork create(data::ColumnDataTypes data_types,
                             std::string target_col, uint32_t n_target_classes,
                             bool use_simpler_model, bool integer_target,
                             char delimiter);

  void index(const dataset::DataSourcePtr& source);

  void clearGraph();

 private:
  // Inherit ModelPipeline constructor privately
  using ModelPipeline::ModelPipeline;
};

using GraphNetworkPtr = std::shared_ptr<GraphNetwork>;

}  // namespace thirdai::automl::models