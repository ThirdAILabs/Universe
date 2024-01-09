#pragma once

#include <bolt/src/nn/tensor/Tensor.h>
#include <bolt/src/train/trainer/Dataset.h>
#include <bolt_vector/src/BoltVector.h>
#include <auto_ml/src/featurization/DataTypes.h>
#include <auto_ml/src/featurization/TabularOptions.h>
#include <dataset/src/dataset_loaders/DatasetLoader.h>
#include <dataset/src/utils/GraphInfo.h>

namespace thirdai::automl {

class GraphDatasetManager {
 public:
  GraphDatasetManager(ColumnDataTypes data_types, std::string target_col,
                      uint32_t n_target_classes, const TabularOptions& options);

  dataset::DatasetLoaderPtr indexAndGetLabeledDatasetLoader(
      const dataset::DataSourcePtr& data_source, bool shuffle,
      dataset::DatasetShuffleConfig shuffle_config =
          dataset::DatasetShuffleConfig());

  void index(const dataset::DataSourcePtr& data_source);

  bolt::TensorList featurizeInput(const dataset::MapInput& input);

  bolt::TensorList featurizeInputBatch(const dataset::MapInputBatch& inputs);

  void clearGraph() { _graph_info->clear(); }

  std::vector<uint32_t> getInputDims() const {
    return _inference_featurizer->getDimensions();
  }

  uint32_t getLabelDim() const { return _n_target_classes; }

  ColumnDataTypes dataTypes() { return _data_types; }

 private:
  ColumnDataTypes _data_types;
  std::string _target_col;
  uint32_t _n_target_classes;
  char _delimiter;
  dataset::TabularFeaturizerPtr _graph_builder, _labeled_featurizer,
      _inference_featurizer;
  GraphInfoPtr _graph_info;

  GraphDatasetManager() {}

  friend cereal::access;

  template <class Archive>
  void serialize(Archive& archive);
};

using GraphDatasetManagerPtr = std::shared_ptr<GraphDatasetManager>;

}  // namespace thirdai::automl