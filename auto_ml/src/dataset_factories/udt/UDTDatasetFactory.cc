#include "UDTDatasetFactory.h"
#include <cereal/archives/binary.hpp>
#include <bolt_vector/src/BoltVector.h>
#include <auto_ml/src/dataset_factories/udt/DataTypes.h>
#include <auto_ml/src/dataset_factories/udt/UDTConfig.h>
#include <dataset/src/DataSource.h>
#include <dataset/src/blocks/BlockInterface.h>
#include <dataset/src/blocks/ColumnNumberMap.h>
#include <dataset/src/blocks/InputTypes.h>
#include <stdexcept>

namespace thirdai::automl::data {

static UDTConfigPtr verifyUDTConfigIsValid(
    const UDTConfigPtr& config,
    const TemporalRelationships& temporal_relationships) {
  FeatureComposer::verifyConfigIsValid(config->data_types, config->target,
                                       temporal_relationships);
  return config;
}

UDTDatasetFactory::UDTDatasetFactory(
    const UDTConfigPtr& config, bool force_parallel,
    uint32_t text_pairgram_word_limit, bool contextual_columns,
    std::optional<dataset::RegressionBinningStrategy> regression_binning)
    : _temporal_relationships(TemporalRelationshipsAutotuner::autotune(
          config->data_types, config->provided_relationships,
          config->lookahead)),
      _config(verifyUDTConfigIsValid(config, _temporal_relationships)),
      _context(std::make_shared<TemporalContext>()),
      _parallel(_temporal_relationships.empty() || force_parallel),
      _text_pairgram_word_limit(text_pairgram_word_limit),
      _contextual_columns(contextual_columns),
      _normalize_target_categories(false),
      _regression_binning(regression_binning),
      _categorical_metadata(_config->data_types, _text_pairgram_word_limit,
                            _contextual_columns, _config->hash_range),
      _labeled_history_updating_processor(makeLabeledUpdatingProcessor()),
      _unlabeled_non_updating_processor(makeUnlabeledNonUpdatingProcessor()) {}

dataset::DatasetLoaderPtr UDTDatasetFactory::getLabeledDatasetLoader(
    dataset::DataSourcePtr data_source, bool training) {
  return std::make_unique<dataset::DatasetLoader>(
      data_source, _labeled_history_updating_processor,
      /* shuffle= */ training);
}

uint32_t UDTDatasetFactory::labelToNeuronId(
    std::variant<uint32_t, std::string> label) {
  if (std::holds_alternative<uint32_t>(label)) {
    if (!_config->integer_target) {
      throw std::invalid_argument(
          "Received an integer but integer_target is set to False (it is "
          "False by default). Target must be passed "
          "in as a string.");
    }
    return std::get<uint32_t>(label);
  }

  const std::string& label_str = std::get<std::string>(label);

  if (_config->integer_target) {
    throw std::invalid_argument(
        "Received a string but integer_target is set to True. Target must be "
        "passed in as "
        "an integer.");
  }

  if (!_vocabs.count(_config->target)) {
    throw std::invalid_argument(
        "Attempted to get label to neuron id map before training.");
  }
  return _vocabs.at(_config->target)->getUid(label_str);
}

std::string UDTDatasetFactory::className(uint32_t neuron_id) const {
  if (_config->integer_target) {
    return std::to_string(neuron_id);
  }
  if (!_vocabs.count(_config->target)) {
    throw std::invalid_argument(
        "Attempted to get id to label map before training.");
  }
  return _vocabs.at(_config->target)->getString(neuron_id);
}

void UDTDatasetFactory::updateMetadata(const std::string& col_name,
                                       const MapInput& update) {
  _categorical_metadata.updateMetadata(col_name, update);
}

void UDTDatasetFactory::updateMetadataBatch(const std::string& col_name,
                                            const MapInputBatch& updates) {
  _categorical_metadata.updateMetadataBatch(col_name, updates);
}

dataset::TabularFeaturizerPtr
UDTDatasetFactory::makeLabeledUpdatingProcessor() {
  if (!_config->data_types.count(_config->target)) {
    throw std::invalid_argument(
        "data_types parameter must include the target column.");
  }

  dataset::BlockPtr label_block = getLabelBlock();

  auto input_blocks = buildInputBlocks(/* should_update_history= */ true);

  auto processor = dataset::TabularFeaturizer::make(
      std::move(input_blocks), {label_block}, /* has_header= */ true,
      /* delimiter= */ _config->delimiter, /* parallel= */ _parallel,
      /* hash_range= */ _config->hash_range);
  return processor;
}

dataset::BlockPtr UDTDatasetFactory::getLabelBlock() {
  auto target_type = _config->data_types.at(_config->target);

  if (asCategorical(target_type)) {
    auto target_config = asCategorical(target_type);
    if (!_config->n_target_classes) {
      throw std::invalid_argument(
          "n_target_classes must be specified for a classification task.");
    }
    if (_config->integer_target) {
      return dataset::NumericalCategoricalBlock::make(
          /* col= */ _config->target,
          /* n_classes= */ _config->n_target_classes.value(),
          /* delimiter= */ target_config->delimiter,
          /* normalize_categories= */ _normalize_target_categories);
    }
    if (!_vocabs.count(_config->target)) {
      _vocabs[_config->target] = dataset::ThreadSafeVocabulary::make(
          /* vocab_size= */ _config->n_target_classes.value());
    }
    return dataset::StringLookupCategoricalBlock::make(
        /* col= */ _config->target, /* vocab= */ _vocabs.at(_config->target),
        /* delimiter= */ target_config->delimiter,
        /* normalize_categories= */ _normalize_target_categories);
  }
  if (asNumerical(target_type)) {
    if (!_regression_binning) {
      throw std::logic_error(
          "Regression binning must be set for numerical outputs.");
    }

    return dataset::RegressionCategoricalBlock::make(
        /* col= */ _config->target,
        /* binning_strategy= */ _regression_binning.value(),
        /* correct_label_radius= */
        UDTConfig::REGRESSION_CORRECT_LABEL_RADIUS,
        /* labels_sum_to_one= */ true);
  }
  throw std::invalid_argument(
      "Target column must have type numerical or categorical.");
}

std::vector<dataset::BlockPtr> UDTDatasetFactory::buildInputBlocks(
    bool should_update_history) {
  std::vector<dataset::BlockPtr> blocks =
      FeatureComposer::makeNonTemporalFeatureBlocks(
          _config->data_types, _config->target, _temporal_relationships,
          _categorical_metadata.metadataVectors(), _text_pairgram_word_limit,
          _contextual_columns);

  if (_temporal_relationships.empty()) {
    return blocks;
  }

  auto temporal_feature_blocks = FeatureComposer::makeTemporalFeatureBlocks(
      *_config, _temporal_relationships,
      _categorical_metadata.metadataVectors(), *_context,
      should_update_history);

  blocks.insert(blocks.end(), temporal_feature_blocks.begin(),
                temporal_feature_blocks.end());
  return blocks;
}

void UDTDatasetFactory::save(const std::string& filename) const {
  std::ofstream filestream =
      dataset::SafeFileIO::ofstream(filename, std::ios::binary);
  save_stream(filestream);
}

void UDTDatasetFactory::save_stream(std::ostream& output_stream) const {
  cereal::BinaryOutputArchive oarchive(output_stream);
  oarchive(*this);
}

UDTDatasetFactoryPtr UDTDatasetFactory::load(const std::string& filename) {
  std::ifstream filestream =
      dataset::SafeFileIO::ifstream(filename, std::ios::binary);
  return load_stream(filestream);
}

UDTDatasetFactoryPtr UDTDatasetFactory::load_stream(
    std::istream& input_stream) {
  cereal::BinaryInputArchive iarchive(input_stream);
  std::shared_ptr<UDTDatasetFactory> deserialize_into(new UDTDatasetFactory());
  iarchive(*deserialize_into);
  return deserialize_into;
}

void UDTDatasetFactory::verifyCanDistribute() {
  auto target_type = _config->data_types.at(_config->target);
  if (asCategorical(target_type) && !_config->integer_target) {
    throw std::invalid_argument(
        "UDT with categorical target without integer_target=True cannot be "
        "trained in distributed "
        "setting. Please convert the categorical target column into "
        "integer target to train UDT in distributed setting.");
  }
  if (!_temporal_relationships.empty()) {
    throw std::invalid_argument(
        "UDT with temporal relationships cannot be trained in a distributed "
        "setting.");
  }
}

}  // namespace thirdai::automl::data

CEREAL_REGISTER_TYPE(thirdai::automl::data::UDTDatasetFactory)