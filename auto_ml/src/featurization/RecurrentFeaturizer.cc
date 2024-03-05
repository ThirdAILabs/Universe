#include "RecurrentFeaturizer.h"
#include <cereal/archives/binary.hpp>
#include <cereal/types/memory.hpp>
#include <cereal/types/vector.hpp>
#include <archive/src/Map.h>
#include <auto_ml/src/featurization/ReservedColumns.h>
#include <auto_ml/src/featurization/TabularTransformations.h>
#include <data/src/transformations/EncodePosition.h>
#include <data/src/transformations/FeatureHash.h>
#include <data/src/transformations/Pipeline.h>
#include <data/src/transformations/Recurrence.h>
#include <data/src/transformations/StringIDLookup.h>
#include <cstddef>
#include <memory>
#include <optional>
#include <stdexcept>

namespace thirdai::automl {

RecurrentFeaturizer::RecurrentFeaturizer(const ColumnDataTypes& data_types,
                                         const std::string& target_name,
                                         const SequenceDataTypePtr& target,
                                         uint32_t n_target_classes,
                                         const TabularOptions& tabular_options)
    : _delimiter(tabular_options.delimiter),
      _state(std::make_shared<data::State>()) {
  if (!target->max_length) {
    throw std::invalid_argument(
        "Paramter max_length must be specified for target sequence.");
  }

  _recurrence_augmentation = std::make_shared<data::Recurrence>(
      /* source_input_column= */ RECURRENT_SEQUENCE,
      /* target_input_column= */ target_name,
      /* source_output_column= */ RECURRENT_SEQUENCE,
      /* target_output_column= */ FEATURIZED_LABELS, n_target_classes,
      target->max_length.value());

  std::tie(_augmenting_transform, _recurrence_augmentation) =
      makeTransformation(data_types, target_name, target, n_target_classes,
                         tabular_options, /*add_recurrence_augmentation=*/true);

  _non_augmenting_transform =
      makeTransformation(data_types, target_name, target, n_target_classes,
                         tabular_options, /*add_recurrence_augmentation=*/false)
          .first;

  _bolt_input_columns = {
      data::OutputColumns::sparse(FEATURIZED_INDICES, FEATURIZED_VALUES)};
  _bolt_label_columns = {data::OutputColumns::sparse(FEATURIZED_LABELS)};
}

std::pair<data::TransformationPtr, std::shared_ptr<data::Recurrence>>
RecurrentFeaturizer::makeTransformation(
    const ColumnDataTypes& data_types, const std::string& target_name,
    const SequenceDataTypePtr& target, uint32_t n_target_classes,
    const TabularOptions& tabular_options,
    bool add_recurrence_augmentation) const {
  auto [input_transforms, outputs] =
      nonTemporalTransformations(data_types, target_name, tabular_options);

  auto target_lookup = std::make_shared<data::StringIDLookup>(
      /* input_column= */ target_name, /* output_column= */ target_name,
      /* vocab_key= */ TARGET_VOCAB, /* max_vocab_size= */ n_target_classes,
      target->delimiter);
  input_transforms.push_back(target_lookup);

  auto target_encoding = std::make_shared<data::OffsetPositionTransform>(
      target_name, RECURRENT_SEQUENCE, target->max_length.value());
  input_transforms.push_back(target_encoding);
  outputs.push_back(RECURRENT_SEQUENCE);

  std::shared_ptr<data::Recurrence> recurrence = nullptr;
  if (add_recurrence_augmentation) {
    recurrence = std::make_shared<data::Recurrence>(
        /* source_input_column= */ RECURRENT_SEQUENCE,
        /* target_input_column= */ target_name,
        /* source_output_column= */ RECURRENT_SEQUENCE,
        /* target_output_column= */ FEATURIZED_LABELS, n_target_classes,
        target->max_length.value());
    input_transforms.push_back(recurrence);
  }

  auto fh = std::make_shared<data::FeatureHash>(
      outputs, FEATURIZED_INDICES, FEATURIZED_VALUES,
      tabular_options.feature_hash_range);
  input_transforms.push_back(fh);

  return {data::Pipeline::make(input_transforms), recurrence};
}

data::LoaderPtr RecurrentFeaturizer::getDataLoader(
    const dataset::DataSourcePtr& data_source, size_t batch_size, bool shuffle,
    bool verbose, dataset::DatasetShuffleConfig shuffle_config) {
  data_source->restart();

  auto data_iter = data::CsvIterator::make(data_source, _delimiter);

  return data::Loader::make(
      data_iter, _augmenting_transform, _state, _bolt_input_columns,
      _bolt_label_columns, /* batch_size= */ batch_size, /* shuffle= */ shuffle,
      /* verbose= */ verbose,
      /* shuffle_buffer_size= */ shuffle_config.min_buffer_size,
      /* shuffle_seed= */ shuffle_config.seed);
}

bolt::TensorList RecurrentFeaturizer::featurizeInput(const MapInput& sample) {
  auto columns = data::ColumnMap::fromMapInput(sample);

  columns = _non_augmenting_transform->apply(std::move(columns), *_state);

  return data::toTensors(columns, _bolt_input_columns);
}

bolt::TensorList RecurrentFeaturizer::featurizeInputBatch(
    const MapInputBatch& samples) {
  auto columns = data::ColumnMap::fromMapInputBatch(samples);

  columns = _non_augmenting_transform->apply(std::move(columns), *_state);

  return data::toTensors(columns, _bolt_input_columns);
}

const data::ThreadSafeVocabularyPtr& RecurrentFeaturizer::vocab() const {
  return _state->getVocab(TARGET_VOCAB);
}

ar::ConstArchivePtr RecurrentFeaturizer::toArchive() const {
  auto map = ar::Map::make();

  map->set("augmenting_transform", _augmenting_transform->toArchive());
  map->set("non_augmenting_transform", _non_augmenting_transform->toArchive());
  map->set("recurrence_augmentation", _recurrence_augmentation->toArchive());

  map->set("bolt_input_columns",
           data::outputColumnsToArchive(_bolt_input_columns));
  map->set("bolt_label_columns",
           data::outputColumnsToArchive(_bolt_label_columns));

  map->set("delimiter", ar::character(_delimiter));

  map->set("state", _state->toArchive());

  return map;
}

std::shared_ptr<RecurrentFeaturizer> RecurrentFeaturizer::fromArchive(
    const ar::Archive& archive) {
  return std::make_shared<RecurrentFeaturizer>(archive);
}

RecurrentFeaturizer::RecurrentFeaturizer(const ar::Archive& archive)
    : _augmenting_transform(data::Transformation::fromArchive(
          *archive.get("augmenting_transform"))),
      _non_augmenting_transform(data::Transformation::fromArchive(
          *archive.get("non_augmenting_transform"))),
      _recurrence_augmentation(std::make_shared<data::Recurrence>(
          *archive.get("recurrence_augmentation"))),
      _bolt_input_columns(
          data::outputColumnsFromArchive(*archive.get("bolt_input_columns"))),
      _bolt_label_columns(
          data::outputColumnsFromArchive(*archive.get("bolt_label_columns"))),
      _delimiter(archive.getAs<ar::Char>("delimiter")),
      _state(data::State::fromArchive(*archive.get("state"))) {}

template void RecurrentFeaturizer::serialize(cereal::BinaryInputArchive&);
template void RecurrentFeaturizer::serialize(cereal::BinaryOutputArchive&);

template <class Archive>
void RecurrentFeaturizer::serialize(Archive& archive) {
  archive(_augmenting_transform, _non_augmenting_transform,
          _recurrence_augmentation, _bolt_input_columns, _bolt_label_columns,
          _delimiter, _state);
}

}  // namespace thirdai::automl