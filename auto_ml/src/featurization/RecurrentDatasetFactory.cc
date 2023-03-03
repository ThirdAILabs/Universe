#include "RecurrentDatasetFactory.h"
#include "TabularBlockComposer.h"
#include <auto_ml/src/dataset_factories/udt/DataTypes.h>
#include <dataset/src/blocks/Augmentation.h>
#include <dataset/src/blocks/BlockInterface.h>
#include <dataset/src/blocks/Categorical.h>
#include <dataset/src/blocks/InputTypes.h>
#include <dataset/src/blocks/RecurrenceAugmentation.h>
#include <utils/StringManipulation.h>
#include <algorithm>
#include <optional>
#include <stdexcept>
#include <vector>

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

RecurrentDatasetFactory::RecurrentDatasetFactory(
    const ColumnDataTypes& data_types, const std::string& target_name,
    const data::SequenceDataTypePtr& target, uint32_t n_target_classes,
    const TabularOptions& tabular_options)
    : _delimiter(tabular_options.delimiter),
      _target_name(target_name),
      _target(target) {
  validate(data_types, target);

  auto labeled_featurizer_input_blocks = makeNonTemporalInputBlocks(
      data_types, /* label_col_names= */ {target_name},
      /* temporal_relationships= */ {},
      /* vectors_map= */ {}, tabular_options);

  // This copies
  auto inference_featurizer_input_blocks = labeled_featurizer_input_blocks;

  // NOLINTNEXTLINE
  _augmentation = dataset::RecurrenceAugmentation::make(
      /* sequence_column= */ target_name,
      /* delimiter= */ _target->delimiter,
      /* max_recurrence= */ _target->max_length.value(),
      /* vocab_size= */ n_target_classes, /* input_vector_index= */ 0,
      /* label_vector_index= */ 1);

  labeled_featurizer_input_blocks.push_back(_augmentation->inputBlock());

  inference_featurizer_input_blocks.push_back(
      dataset::NumericalCategoricalBlock::make(
          /* col= */ target_name,
          /* n_classes= */ _target->max_length.value() * (n_target_classes + 1),
          /* delimiter= */ _target->delimiter));

  _labeled_featurizer = dataset::TabularFeaturizer::make(
      /* block_lists= */
      {
          dataset::BlockList(
              std::move(labeled_featurizer_input_blocks),
              /* hash_range= */ tabular_options.feature_hash_range),
          dataset::BlockList({_augmentation->labelBlock()}),
      },
      /* augmentation= */ _augmentation,
      /* has_header= */ true, _delimiter, /* parallel= */ true);

  _inference_featurizer = dataset::TabularFeaturizer::make(
      /* block_lists= */
      {
          dataset::BlockList(
              std::move(inference_featurizer_input_blocks),
              /* hash_range= */ tabular_options.feature_hash_range),
      },
      /* has_header= */ true, _delimiter, /* parallel= */ true);
}

dataset::DatasetLoaderPtr RecurrentDatasetFactory::getDatasetLoader(
    const dataset::DataSourcePtr& data_source, bool training) {
  return std::make_unique<dataset::DatasetLoader>(data_source,
                                                  _labeled_featurizer,
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

uint32_t RecurrentDatasetFactory::elementIdAtStep(const BoltVector& output,
                                                  uint32_t step) {
  return _augmentation->elementIdAtStep(output, step);
}

std::string RecurrentDatasetFactory::elementString(uint32_t element_id) {
  return _augmentation->elementString(element_id);
}

bool RecurrentDatasetFactory::isEOS(uint32_t element_id) {
  return _augmentation->isEOS(element_id);
}

void RecurrentDatasetFactory::addPredictionToSample(MapInput& sample,
                                                    uint32_t prediction) {
  auto& intermediate_column = sample[_target_name];
  if (!intermediate_column.empty()) {
    intermediate_column += _target->delimiter;
  }
  intermediate_column += std::to_string(prediction);
}

template void RecurrentDatasetFactory::serialize(cereal::BinaryInputArchive&);
template void RecurrentDatasetFactory::serialize(cereal::BinaryOutputArchive&);

template <class Archive>
void RecurrentDatasetFactory::serialize(Archive& archive) {
  archive(_delimiter, _target_name, _target, _augmentation, _labeled_featurizer,
          _inference_featurizer);
}

}  // namespace thirdai::automl::data