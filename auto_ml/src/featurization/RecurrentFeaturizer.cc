#include "RecurrentFeaturizer.h"
#include <cereal/archives/binary.hpp>
#include <cereal/types/memory.hpp>
#include <cereal/types/vector.hpp>
#include <auto_ml/src/featurization/ReservedColumns.h>
#include <auto_ml/src/featurization/TabularTransformations.h>
#include <data/src/transformations/EncodePosition.h>
#include <data/src/transformations/FeatureHash.h>
#include <data/src/transformations/StringIDLookup.h>
#include <data/src/transformations/TransformationList.h>
#include <data/src/transformations/UnrollSequence.h>
#include <optional>

namespace thirdai::automl {

RecurrentFeaturizer::RecurrentFeaturizer(
    const data::ColumnDataTypes& data_types, const std::string& target_name,
    const data::SequenceDataTypePtr& target, uint32_t n_target_classes,
    const data::TabularOptions& tabular_options)
    : _delimiter(tabular_options.delimiter),
      _state(std::make_shared<thirdai::data::State>()) {
  auto [input_transforms, outputs] =
      nonTemporalTransformations(data_types, target_name, tabular_options);

  auto target_lookup = std::make_shared<thirdai::data::StringIDLookup>(
      /* input_column= */ target_name, /* output_column= */ target_name,
      /* vocab_key= */ TARGET_VOCAB, /* max_vocab_size= */ n_target_classes,
      target->delimiter);

  auto target_encoding =
      std::make_shared<thirdai::data::OffsetPositionTransform>(
          target_name, RECURRENT_SEQUENCE, target->max_length.value());

  input_transforms.push_back(target_lookup);
  input_transforms.push_back(target_encoding);

  outputs.push_back(RECURRENT_SEQUENCE);

  auto fh = std::make_shared<thirdai::data::FeatureHash>(
      outputs, FEATURE_HASH_INDICES, FEATURE_HASH_VALUES,
      tabular_options.feature_hash_range);

  input_transforms.push_back(fh);

  _recurrence_augmentation = std::make_shared<thirdai::data::UnrollSequence>(
      /* source_input_column= */ RECURRENT_SEQUENCE,
      /* target_input_column= */ RECURRENT_SEQUENCE,
      /* source_output_column= */ RECURRENT_SEQUENCE,
      /* target_output_column= */ FEATURIZED_LABELS);

  _input_transform = thirdai::data::TransformationList::make(input_transforms);

  _bolt_input_columns = {
      thirdai::data::OutputColumns(FEATURE_HASH_INDICES, FEATURE_HASH_VALUES)};
  _bolt_label_columns = {thirdai::data::OutputColumns(FEATURIZED_LABELS)};
}

thirdai::data::LoaderPtr RecurrentFeaturizer::getDataLoader(
    const dataset::DataSourcePtr& data_source, size_t batch_size, bool shuffle,
    bool verbose, dataset::DatasetShuffleConfig shuffle_config) {
  auto csv_data_source = dataset::CsvDataSource::make(data_source, _delimiter);

  csv_data_source->restart();

  thirdai::data::ColumnMapIterator data_iter(csv_data_source, _delimiter);

  auto transformation_list = thirdai::data::TransformationList::make(
      {_input_transform, _recurrence_augmentation});

  return thirdai::data::Loader::make(
      data_iter, transformation_list, _state, _bolt_input_columns,
      _bolt_label_columns, /* batch_size= */ batch_size, /* shuffle= */ shuffle,
      /* verbose= */ verbose,
      /* shuffle_buffer_size= */ shuffle_config.min_buffer_size,
      /* shuffle_seed= */ shuffle_config.seed);
}

bolt::TensorList RecurrentFeaturizer::featurizeInput(const MapInput& sample) {
  auto columns = thirdai::data::ColumnMap::fromMapInput(sample);

  columns = _input_transform->apply(columns, *_state);

  return thirdai::data::toTensors(columns, _bolt_input_columns);
}

bolt::TensorList RecurrentFeaturizer::featurizeInputBatch(
    const MapInputBatch& samples) {
  auto columns = thirdai::data::ColumnMap::fromMapInputBatch(samples);

  columns = _input_transform->apply(columns, *_state);

  return thirdai::data::toTensors(columns, _bolt_input_columns);
}

const thirdai::data::ThreadSafeVocabularyPtr& RecurrentFeaturizer::vocab()
    const {
  return _state->getVocab(TARGET_VOCAB);
}

template void RecurrentFeaturizer::serialize(cereal::BinaryInputArchive&);
template void RecurrentFeaturizer::serialize(cereal::BinaryOutputArchive&);

template <class Archive>
void RecurrentFeaturizer::serialize(Archive& archive) {
  archive(_input_transform, _recurrence_augmentation, _bolt_input_columns,
          _bolt_label_columns, _delimiter, _state);
}

}  // namespace thirdai::automl