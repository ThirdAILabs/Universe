#include "RecurrentFeaturizer.h"
#include <auto_ml/src/featurization/TabularTransformations.h>
#include <data/src/transformations/EncodePosition.h>
#include <data/src/transformations/FeatureHash.h>
#include <data/src/transformations/StringIDLookup.h>
#include <data/src/transformations/TransformationList.h>
#include <data/src/transformations/UnrollSequence.h>
#include <optional>

namespace thirdai::automl {

static const std::string VOCAB = "__recurrent_vocab__";

RecurrentFeaturizer::RecurrentFeaturizer(
    const data::ColumnDataTypes& data_types, const std::string& target_name,
    const data::SequenceDataTypePtr& target, uint32_t n_target_classes,
    const data::TabularOptions& tabular_options)
    : _delimiter(tabular_options.delimiter),
      _state(std::make_shared<thirdai::data::State>()) {
  auto [input_transforms, outputs] =
      nonTemporalTransformations(data_types, target_name, tabular_options);

  auto target_lookup = std::make_shared<thirdai::data::StringIDLookup>(
      target_name, target_name, target_name, n_target_classes,
      target->delimiter);

  auto target_encoding =
      std::make_shared<thirdai::data::OffsetPositionTransform>(
          target_name, "__target_sequence__", target->max_length.value());

  input_transforms.push_back(target_lookup);
  input_transforms.push_back(target_encoding);

  outputs.push_back("__target_sequence__");

  auto fh = std::make_shared<thirdai::data::FeatureHash>(
      outputs, "__featurized_indices__", "__featurized_values__",
      tabular_options.feature_hash_range);

  input_transforms.push_back(fh);

  _recurrence_augmentation = std::make_shared<thirdai::data::UnrollSequence>(
      "__target_sequence__", "__target_sequence__", "__target_sequence__",
      "__next_tokens__");

  _input_transform = thirdai::data::TransformationList::make(input_transforms);

  _bolt_input_columns = {{"__featurized_indices__", "__featurized_values__"}};
  _bolt_label_columns = {{"__next_tokens__", std::nullopt}};
}

thirdai::data::LoaderPtr RecurrentFeaturizer::getDataLoader(
    const dataset::DataSourcePtr& data_source, size_t batch_size, bool shuffle,
    bool verbose, dataset::DatasetShuffleConfig shuffle_config) {
  auto csv_data_source = dataset::CsvDataSource::make(data_source, _delimiter);

  data_source->restart();

  thirdai::data::ColumnMapIterator data_iter(data_source, _delimiter);

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
  return _state->getVocab(VOCAB);
}

}  // namespace thirdai::automl