#pragma once

#include <bolt_vector/src/BoltVector.h>
#include <auto_ml/src/Aliases.h>
#include <auto_ml/src/dataset_factories/udt/DataTypes.h>
#include <auto_ml/src/featurization/TabularOptions.h>
#include <dataset/src/DataSource.h>
#include <dataset/src/blocks/Sequence.h>
#include <dataset/src/dataset_loaders/DatasetLoader.h>
#include <dataset/src/featurizers/TabularFeaturizer.h>
#include <memory>
#include <vector>

namespace thirdai::automl::data {

class RecurrentDatasetFactory {
 public:
  RecurrentDatasetFactory(ColumnDataTypes data_types,
                          const std::string& target_name,
                          const data::SequenceDataTypePtr& target,
                          uint32_t n_target_classes,
                          const TabularOptions& tabular_options);

  uint32_t outputDim() { return _sequence_target_block->featureDim(); }

  dataset::DatasetLoaderPtr getDatasetLoader(dataset::DataSourcePtr data_source,
                                             bool training);

  std::vector<BoltVector> featurizeInput(const MapInput& sample);

  std::vector<BoltBatch> featurizeInputBatch(const MapInputBatch& samples);

  std::string classNameAtStep(const BoltVector& output, uint32_t step);

  void addPredictionToSample(MapInput& sample, const std::string& prediction);

  std::string stitchTargetSequence(const std::vector<std::string>& predictions);

 private:
  RecurrentDatasetFactory() {}

  friend cereal::access;

  template <class Archive>
  void serialize(Archive& archive);

  char _delimiter;
  SequenceDataTypePtr _target;

  std::string _intermediate_column;
  std::string _current_step_target_column;
  std::string _step_column;

  dataset::SequenceTargetBlockPtr _sequence_target_block;

  dataset::TabularFeaturizerPtr _labeled_featurizer;
  dataset::TabularFeaturizerPtr _inference_featurizer;
};

using RecurrentDatasetFactoryPtr = std::shared_ptr<RecurrentDatasetFactory>;

}  // namespace thirdai::automl::data