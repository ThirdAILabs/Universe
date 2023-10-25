#include "RecurrentFeaturizer.h"
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
      data::OutputColumns(FEATURIZED_INDICES, FEATURIZED_VALUES)};
  _bolt_label_columns = {data::OutputColumns(FEATURIZED_LABELS)};
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

RecurrentFeaturizer::RecurrentFeaturizer(
    const proto::udt::RecurrentFeaturizer& featurizer)
    : _augmenting_transform(
          data::Transformation::fromProto(featurizer.augmenting_transform())),
      _non_augmenting_transform(data::Transformation::fromProto(
          featurizer.non_augmenting_transform())),
      _bolt_input_columns(thirdai::data::outputColumnsListFromProto(
          featurizer.bolt_input_columns())),
      _bolt_label_columns(thirdai::data::outputColumnsListFromProto(
          featurizer.bolt_label_columns())),
      _delimiter(featurizer.delimiter()),
      _state(thirdai::data::State::fromProto(featurizer.state())) {
  auto augmentation =
      data::Transformation::fromProto(featurizer.recurrence_augmentation());

  // The toProto method on the recurrent transformation returns a Transformation
  // proto object, and when we invoke Transformation::fromProto we get an
  // instance of the Transformation base class, thus we need to downcast it to
  // get the recurrence augmentation.
  _recurrence_augmentation =
      std::dynamic_pointer_cast<thirdai::data::Recurrence>(augmentation);
  if (!_recurrence_augmentation) {
    throw std::invalid_argument(
        "Expected recurrence augmentation transformation in fromProto.");
  }
}

data::LoaderPtr RecurrentFeaturizer::getDataLoader(
    const dataset::DataSourcePtr& data_source, size_t batch_size, bool shuffle,
    bool verbose, dataset::DatasetShuffleConfig shuffle_config) {
  auto csv_data_source = dataset::CsvDataSource::make(data_source, _delimiter);

  csv_data_source->restart();

  auto data_iter = data::CsvIterator::make(csv_data_source, _delimiter);

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

proto::udt::RecurrentFeaturizer* RecurrentFeaturizer::toProto() const {
  auto* featurizer = new proto::udt::RecurrentFeaturizer();

  featurizer->set_allocated_augmenting_transform(
      _augmenting_transform->toProto());
  featurizer->set_allocated_non_augmenting_transform(
      _non_augmenting_transform->toProto());
  featurizer->set_allocated_recurrence_augmentation(
      _recurrence_augmentation->toProto());

  featurizer->set_allocated_bolt_input_columns(
      thirdai::data::outputColumnsListToProto(_bolt_input_columns));
  featurizer->set_allocated_bolt_label_columns(
      thirdai::data::outputColumnsListToProto(_bolt_label_columns));

  featurizer->set_delimiter(_delimiter);

  featurizer->set_allocated_state(_state->toProto());

  return featurizer;
}

}  // namespace thirdai::automl