#include "RecurrentDatasetFactory.h"
#include "TabularBlockComposer.h"
#include <auto_ml/src/dataset_factories/udt/DataTypes.h>
#include <dataset/src/RecursionWrapper.h>
#include <dataset/src/blocks/InputTypes.h>
#include <utils/StringManipulation.h>
#include <stdexcept>

namespace thirdai::automl::data {

static void validate(const ColumnDataTypes& data_types,
                     const data::SequenceDataTypePtr& target) {
  for (const auto& [name, type] : data_types) {
    if (auto categorical = asCategorical(type)) {
      if (categorical->metadata_config) {
        throw std::invalid_argument(
            "UDT does not support categorical metadata when doing recurrent "
            "classification.");
      }
    }
  }

  if (!target->max_length) {
    throw std::invalid_argument("Must provide max_length for target sequence.");
  }
}

// Augments a proposed name until it is unique in the data_types map.
static std::string uniqueName(const ColumnDataTypes& data_types,
                              std::string proposed_name) {
  while (data_types.count(proposed_name)) {
    // Use uncommon character '$' to avoid colliding with other column names;
    proposed_name += "$";
  }
  return proposed_name;
}

RecurrentDatasetFactory::RecurrentDatasetFactory(
    ColumnDataTypes data_types, const std::string& target_name,
    const data::SequenceDataTypePtr& target, uint32_t n_target_classes,
    const TabularOptions& tabular_options)
    : _delimiter(tabular_options.delimiter),
      _target(target),
      _current_step_target_column(target_name) {
  validate(data_types, target);

  // Increment by 1 for EOS token
  n_target_classes++;

  // This column will contain the predicted target sequence so far.
  // The target column will contain the single next item to be predicted.
  _intermediate_column = uniqueName(data_types, target_name + "_inter");
  data_types[_intermediate_column] = target;

  // This column will contain the current prediction step.
  // Only used by SequenceTargetBlock. Input vectors will not encode this.
  _step_column = uniqueName(data_types, target_name + "_step");

  auto input_blocks = makeNonTemporalInputBlocks(
      data_types, /* label_col_names= */ {target_name},
      /* temporal_relationships= */ {},
      /* vectors_map= */ {}, tabular_options);

  _sequence_target_block = dataset::SequenceTargetBlock::make(
      /* target_col= */ target_name, /* step_col= */ _step_column,
      /* max_steps= */ target->max_length.value(),
      /* vocabulary_size= */ n_target_classes);

  _labeled_featurizer = dataset::TabularFeaturizer::make(
      input_blocks, {_sequence_target_block},
      /* has_header= */ true, _delimiter, /* parallel= */ true,
      /* hash_range= */ tabular_options.feature_hash_range);

  _inference_featurizer = dataset::TabularFeaturizer::make(
      input_blocks, /* label_blocks= */ {},
      /* has_header= */ true, _delimiter,
      /* parallel= */ true,
      /* hash_range= */ tabular_options.feature_hash_range);
}

dataset::DatasetLoaderPtr RecurrentDatasetFactory::getDatasetLoader(
    dataset::DataSourcePtr data_source, bool training) {
  auto source = dataset::RecursionWrapper::make(
      /* source= */ std::move(data_source),
      /* column_delimiter= */ _delimiter,
      /* target_delimiter= */ _target->delimiter,
      /* intermediate_column= */ _intermediate_column,
      /* target_column= */ _current_step_target_column,
      /* step_column= */ _step_column,
      /* max_recursion_depth= */ _target->max_length.value());
  return std::make_unique<dataset::DatasetLoader>(source, _labeled_featurizer,
                                                  /* shuffle= */ training);
}

std::vector<BoltVector> RecurrentDatasetFactory::featurizeInput(
    const MapInput& sample) {
  dataset::MapSampleRef sample_ref(sample);
  return {_inference_featurizer->makeInputVector(sample_ref)};
}

std::vector<BoltBatch> RecurrentDatasetFactory::featurizeInputBatch(
    const MapInputBatch& samples) {
  dataset::MapBatchRef batch_ref(samples);
  auto batches = _inference_featurizer->featurize(batch_ref);

  // We cannot use the initializer list because the copy constructor is
  // deleted for BoltBatch.
  std::vector<BoltBatch> batch_list;
  batch_list.emplace_back(std::move(batches.at(0)));
  return batch_list;
}

std::string RecurrentDatasetFactory::classNameAtStep(const BoltVector& output,
                                                     uint32_t step) {
  return _sequence_target_block->classNameAtStep(output, step);
}

void RecurrentDatasetFactory::addPredictionToSample(
    MapInput& sample, const std::string& prediction) {
  auto& intermediate_column = sample[_intermediate_column];
  if (!intermediate_column.empty()) {
    intermediate_column += _target->delimiter;
  }
  intermediate_column += prediction;
}

std::string RecurrentDatasetFactory::stitchTargetSequence(
    const std::vector<std::string>& predictions) {
  return text::join(predictions, /* delimiter= */ {_target->delimiter});
}

template void RecurrentDatasetFactory::serialize(cereal::BinaryInputArchive&);
template void RecurrentDatasetFactory::serialize(cereal::BinaryOutputArchive&);

template <class Archive>
void RecurrentDatasetFactory::serialize(Archive& archive) {
  archive(_delimiter, _target, _intermediate_column,
          _current_step_target_column, _step_column, _sequence_target_block,
          _labeled_featurizer, _inference_featurizer);
}

}  // namespace thirdai::automl::data