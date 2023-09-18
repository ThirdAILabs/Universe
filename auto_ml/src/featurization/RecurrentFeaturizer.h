#pragma once

#include <cereal/access.hpp>
#include <auto_ml/src/featurization/DataTypes.h>
#include <auto_ml/src/featurization/TabularOptions.h>
#include <data/src/Loader.h>
#include <data/src/transformations/Recurrence.h>
#include <data/src/transformations/Transformation.h>
#include <dataset/src/dataset_loaders/DatasetLoader.h>

namespace thirdai::automl {

class RecurrentFeaturizer {
 public:
  RecurrentFeaturizer(const data::ColumnDataTypes& data_types,
                      const std::string& target_name,
                      const data::SequenceDataTypePtr& target,
                      uint32_t n_target_classes,
                      const data::TabularOptions& tabular_options);

  thirdai::data::LoaderPtr getDataLoader(
      const dataset::DataSourcePtr& data_source, size_t batch_size,
      bool shuffle, bool verbose,
      dataset::DatasetShuffleConfig shuffle_config =
          dataset::DatasetShuffleConfig());

  bolt::TensorList featurizeInput(const MapInput& sample);

  bolt::TensorList featurizeInputBatch(const MapInputBatch& samples);

  const thirdai::data::ThreadSafeVocabularyPtr& vocab() const;

  bool isEos(uint32_t token) const {
    return _recurrence_augmentation->isEOS(token);
  }

  size_t vocabSize() const {
    return _recurrence_augmentation->totalVocabSize();
  }

  std::pair<uint32_t, uint32_t> outputRangeForStep(uint32_t step) const {
    return _recurrence_augmentation->rangeForStep(step);
  }

  std::string targetTokenToString(uint32_t token) const {
    uint32_t vocab_id = _recurrence_augmentation->toTargetInputToken(token);
    return vocab()->getString(vocab_id);
  }

 private:
  thirdai::data::TransformationPtr makeTransformation(
      const data::ColumnDataTypes& data_types, const std::string& target_name,
      const data::SequenceDataTypePtr& target, uint32_t n_target_classes,
      const data::TabularOptions& tabular_options,
      const thirdai::data::TransformationPtr& augmentation) const;

  thirdai::data::TransformationPtr _augmenting_transform;
  thirdai::data::TransformationPtr _non_augmenting_transform;
  std::shared_ptr<thirdai::data::Recurrence> _recurrence_augmentation;

  thirdai::data::OutputColumnsList _bolt_input_columns;
  thirdai::data::OutputColumnsList _bolt_label_columns;

  char _delimiter;

  thirdai::data::StatePtr _state;

  const std::string TARGET_VOCAB = "__recurrent_vocab__";

  RecurrentFeaturizer() {}

  friend class cereal::access;
  template <class Archive>
  void serialize(Archive& archive);
};

using RecurrentFeaturizerPtr = std::shared_ptr<RecurrentFeaturizer>;

}  // namespace thirdai::automl