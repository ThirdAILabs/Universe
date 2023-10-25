#pragma once

#include <auto_ml/src/featurization/DataTypes.h>
#include <auto_ml/src/featurization/TabularOptions.h>
#include <data/src/Loader.h>
#include <data/src/transformations/State.h>
#include <data/src/transformations/Transformation.h>
#include <dataset/src/dataset_loaders/DatasetLoader.h>
#include <dataset/src/utils/GraphInfo.h>
#include <proto/featurizers.pb.h>

namespace thirdai::automl {

class GraphFeaturizer {
 public:
  GraphFeaturizer(const ColumnDataTypes& data_types,
                  const std::string& target_col, uint32_t n_target_classes,
                  const TabularOptions& options);

  explicit GraphFeaturizer(const proto::udt::GraphFeaturizer& featurizer);

  data::LoaderPtr indexAndGetDataLoader(
      const dataset::DataSourcePtr& data_source, size_t batch_size,
      bool shuffle, bool verbose,
      dataset::DatasetShuffleConfig shuffle_config =
          dataset::DatasetShuffleConfig());

  void index(const dataset::DataSourcePtr& data_source);

  bolt::TensorList featurizeInput(const MapInput& sample);

  bolt::TensorList featurizeInputBatch(const MapInputBatch& samples);

  void clearGraph() { _state->graph()->clear(); }

  proto::udt::GraphFeaturizer* toProto() const;

 private:
  static std::pair<data::TransformationPtr, std::string> nodeId(
      const ColumnDataTypes& data_types);

  static std::pair<data::TransformationPtr, std::string> neighborFeatures(
      const std::string& nod_id_col);

  static std::pair<data::TransformationPtr, std::string> neighborIds(
      const std::string& nod_id_col);

  static std::pair<data::TransformationPtr, GraphInfoPtr> graphBuilder(
      const ColumnDataTypes& data_types);

  data::TransformationPtr _input_transform;
  data::TransformationPtr _label_transform;
  data::TransformationPtr _graph_builder;

  data::OutputColumnsList _bolt_input_columns;
  data::OutputColumnsList _bolt_label_columns;

  char _delimiter;

  thirdai::data::StatePtr _state;
};

using GraphFeaturizerPtr = std::shared_ptr<GraphFeaturizer>;

}  // namespace thirdai::automl