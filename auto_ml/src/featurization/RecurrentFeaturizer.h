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
  RecurrentFeaturizer(const ColumnDataTypes& data_types,
                      const std::string& target_name,
                      const SequenceDataTypePtr& target,
                      uint32_t n_target_classes,
                      const TabularOptions& tabular_options);

  explicit RecurrentFeaturizer(const ar::Archive& archive);

  data::LoaderPtr getDataLoader(const dataset::DataSourcePtr& data_source,
                                size_t batch_size, bool shuffle, bool verbose,
                                dataset::DatasetShuffleConfig shuffle_config =
                                    dataset::DatasetShuffleConfig());

  bolt::TensorList featurizeInput(const MapInput& sample);

  bolt::TensorList featurizeInputBatch(const MapInputBatch& samples);

  const data::ThreadSafeVocabularyPtr& vocab() const;

  bool isEos(uint32_t token) const {
    return _recurrence_augmentation->isEOS(token);
  }

  size_t vocabSize() const {
    return _recurrence_augmentation->totalVocabSize();
  }

  ar::ConstArchivePtr toArchive() const;

  static std::shared_ptr<RecurrentFeaturizer> fromArchive(
      const ar::Archive& archive);

 private:
  std::pair<data::TransformationPtr, std::shared_ptr<data::Recurrence>>
  makeTransformation(const ColumnDataTypes& data_types,
                     const std::string& target_name,
                     const SequenceDataTypePtr& target,
                     uint32_t n_target_classes,
                     const TabularOptions& tabular_options,
                     bool add_recurrence_augmentation) const;

  data::TransformationPtr _augmenting_transform;
  data::TransformationPtr _non_augmenting_transform;
  std::shared_ptr<data::Recurrence> _recurrence_augmentation;

  data::OutputColumnsList _bolt_input_columns;
  data::OutputColumnsList _bolt_label_columns;

  char _delimiter;

  data::StatePtr _state;

  const std::string TARGET_VOCAB = "__recurrent_vocab__";

  RecurrentFeaturizer() {}

  friend class cereal::access;
  template <class Archive>
  void serialize(Archive& archive);
};

using RecurrentFeaturizerPtr = std::shared_ptr<RecurrentFeaturizer>;

}  // namespace thirdai::automl