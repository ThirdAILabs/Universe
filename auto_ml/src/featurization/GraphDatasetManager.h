#pragma once

#include <auto_ml/src/dataset_factories/udt/DataTypes.h>
#include <auto_ml/src/featurization/TabularOptions.h>
#include <dataset/src/dataset_loaders/DatasetLoader.h>
#include <dataset/src/utils/GraphInfo.h>

namespace thirdai::automl::data {

class GraphDatasetManager {
 public:
  GraphDatasetManager(data::ColumnDataTypes data_types, std::string target_col,
                      uint32_t n_target_classes, const TabularOptions& options);

  dataset::DatasetLoaderPtr indexAndGetDatasetLoader(
      const dataset::DataSourcePtr& data_source, bool shuffle);

  void index(const dataset::DataSourcePtr& data_source);

  void clearGraph() { _graph_info->clear(); }

  std::vector<uint32_t> getInputDims() const {
    return _featurizer->getDimensions();
  }

  uint32_t getLabelDim() const { return _n_target_classes; }

 private:
  data::ColumnDataTypes _data_types;
  std::string _target_col;
  uint32_t _n_target_classes;
  char _delimiter;
  dataset::TabularFeaturizerPtr _graph_builder, _featurizer;
  GraphInfoPtr _graph_info;

  GraphDatasetManager() {}

  friend cereal::access;

  template <class Archive>
  void serialize(Archive& archive);
};

using GraphDatasetManagerPtr = std::shared_ptr<GraphDatasetManager>;

}  // namespace thirdai::automl::data