#pragma once

#include <bolt_vector/src/BoltVector.h>
#include <auto_ml/src/Aliases.h>
#include <auto_ml/src/dataset_factories/udt/DataTypes.h>
#include <auto_ml/src/featurization/TabularOptions.h>
#include <dataset/src/DataSource.h>
#include <dataset/src/blocks/RecurrenceAugmentation.h>
#include <dataset/src/blocks/Sequence.h>
#include <dataset/src/dataset_loaders/DatasetLoader.h>
#include <dataset/src/featurizers/TabularFeaturizer.h>
#include <memory>
#include <vector>

namespace thirdai::automl::data {

class RecurrentDatasetFactory {
 public:
  RecurrentDatasetFactory(const ColumnDataTypes& data_types,
                          const std::string& target_name,
                          const data::SequenceDataTypePtr& target,
                          uint32_t n_target_classes,
                          const TabularOptions& tabular_options);

  uint32_t outputDim() { return _labeled_featurizer->getDimensions().back(); }

  dataset::DatasetLoaderPtr getDatasetLoader(
      const dataset::DataSourcePtr& data_source, bool training);

  std::vector<BoltVector> featurizeInput(const MapInput& sample);

  std::vector<BoltBatch> featurizeInputBatch(const MapInputBatch& samples);

  uint32_t elementIdAtStep(const BoltVector& output, uint32_t step);

  std::string elementString(uint32_t element_id);

  bool isEOS(uint32_t element_id);

  void addPredictionToSample(MapInput& sample, uint32_t prediction);

 private:
  RecurrentDatasetFactory() {}

  friend cereal::access;

  template <class Archive>
  void serialize(Archive& archive);

  char _delimiter;
  std::string _target_name;
  SequenceDataTypePtr _target;

  dataset::RecurrenceAugmentationPtr _augmentation;

  dataset::TabularFeaturizerPtr _labeled_featurizer;
  dataset::TabularFeaturizerPtr _inference_featurizer;
};

using RecurrentDatasetFactoryPtr = std::shared_ptr<RecurrentDatasetFactory>;

}  // namespace thirdai::automl::data