#include "UDTDatasetFactory.h"
#include <cereal/archives/binary.hpp>
#include <stdexcept>

namespace thirdai::automl::data {

using dataset::ColumnNumberMap;

DatasetLoaderPtr UDTDatasetFactory::getLabeledDatasetLoader(
    std::shared_ptr<dataset::DataLoader> data_loader, bool training) {
  auto column_number_map =
      makeColumnNumberMapFromHeader(*data_loader, _config->delimiter);

  // The batch processor will treat the next line as a header
  // Restart so batch processor does not skip a sample.
  data_loader->restart();

  _labeled_history_updating_processor->updateColumnNumbers(*column_number_map);
  _unlabeled_non_updating_processor->updateColumnNumbers(*column_number_map);

  return std::make_unique<GenericDatasetLoader>(
      data_loader, _labeled_history_updating_processor,
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

PreprocessedVectorsMap UDTDatasetFactory::processAllMetadata() {
  PreprocessedVectorsMap metadata_vectors;
  for (const auto& [col_name, col_type] : _config->data_types) {
    if (auto categorical = asCategorical(col_type)) {
      if (categorical->metadata_config) {
        metadata_vectors[col_name] =
            makeProcessedVectorsForCategoricalColumn(col_name, categorical);
      }
    }
  }
  return metadata_vectors;
}

dataset::PreprocessedVectorsPtr
UDTDatasetFactory::makeProcessedVectorsForCategoricalColumn(
    const std::string& col_name, const CategoricalDataTypePtr& categorical) {
  if (!categorical->metadata_config) {
    throw std::invalid_argument("The given categorical column (" + col_name +
                                ") does not have a metadata config.");
  }

  auto metadata = categorical->metadata_config;

  auto data_loader =
      dataset::SimpleFileDataLoader::make(metadata->metadata_file,
                                          /* target_batch_size= */ 2048);

  auto column_numbers =
      makeColumnNumberMapFromHeader(*data_loader, metadata->delimiter);

  auto input_blocks = buildMetadataInputBlocks(*metadata);

  auto key_vocab = dataset::ThreadSafeVocabulary::make(
      /* vocab_size= */ 0, /* limit_vocab_size= */ false);
  auto label_block = dataset::StringLookupCategoricalBlock::make(
      column_numbers->at(metadata->key), key_vocab);

  // Here we set parallel=true because there are no temporal
  // relationships in the metadata file.
  dataset::StreamingGenericDatasetLoader metadata_loader(
      /* loader= */ data_loader,
      /* processor= */
      dataset::GenericBatchProcessor::make(
          /* input_blocks= */ std::move(input_blocks),
          /* label_blocks= */ {std::move(label_block)},
          /* has_header= */ false, /* delimiter= */ metadata->delimiter,
          /* parallel= */ true, /* hash_range= */ _config->hash_range));

  return preprocessedVectorsFromDataset(metadata_loader, *key_vocab);
}

ColumnNumberMapPtr UDTDatasetFactory::makeColumnNumberMapFromHeader(
    dataset::DataLoader& data_loader, char delimiter) {
  auto header = data_loader.nextLine();
  if (!header) {
    throw std::invalid_argument(
        "The dataset must have a header that contains column names.");
  }

  return std::make_shared<ColumnNumberMap>(*header, delimiter);
}

std::vector<dataset::BlockPtr> UDTDatasetFactory::buildMetadataInputBlocks(
    const CategoricalMetadataConfig& metadata_config) const {
  UDTConfig feature_config(
      /* data_types= */ metadata_config.column_data_types,
      /* temporal_tracking_relationships= */ {},
      /* target= */ metadata_config.key,
      /* n_target_classes= */ 0);
  TemporalRelationships empty_temporal_relationships;

  PreprocessedVectorsMap empty_vectors_map;

  return FeatureComposer::makeNonTemporalFeatureBlocks(
      feature_config, empty_temporal_relationships, empty_vectors_map,
      _text_pairgram_word_limit, _contextual_columns);
}

dataset::PreprocessedVectorsPtr
UDTDatasetFactory::preprocessedVectorsFromDataset(
    dataset::StreamingGenericDatasetLoader& dataset,
    dataset::ThreadSafeVocabulary& key_vocab) {
  auto [vectors, ids] = dataset.loadInMemory();

  std::unordered_map<std::string, BoltVector> preprocessed_vectors(
      vectors->len());

  for (uint32_t batch = 0; batch < vectors->numBatches(); batch++) {
    for (uint32_t vec = 0; vec < vectors->at(batch).getBatchSize(); vec++) {
      auto id = ids->at(batch)[vec].active_neurons[0];
      auto key = key_vocab.getString(id);
      preprocessed_vectors[key] = std::move(vectors->at(batch)[vec]);
    }
  }

  return std::make_shared<dataset::PreprocessedVectors>(
      std::move(preprocessed_vectors), dataset.getInputDim());
}

dataset::GenericBatchProcessorPtr
UDTDatasetFactory::makeLabeledUpdatingProcessor() {
  if (!_config->data_types.count(_config->target)) {
    throw std::invalid_argument(
        "data_types parameter must include the target column.");
  }

  dataset::BlockPtr label_block = getLabelBlock();

  auto input_blocks = buildInputBlocks(/* should_update_history= */ true);

  auto processor = dataset::GenericBatchProcessor::make(
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
          /* delimiter= */ target_config->delimiter);
    }
    if (!_vocabs.count(_config->target)) {
      _vocabs[_config->target] = dataset::ThreadSafeVocabulary::make(
          /* vocab_size= */ _config->n_target_classes.value());
    }
    return dataset::StringLookupCategoricalBlock::make(
        /* col= */ _config->target, /* vocab= */ _vocabs.at(_config->target),
        /* delimiter= */ target_config->delimiter);
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
          *_config, _temporal_relationships, _vectors_map,
          _text_pairgram_word_limit, _contextual_columns);

  if (_temporal_relationships.empty()) {
    return blocks;
  }

  auto temporal_feature_blocks = FeatureComposer::makeTemporalFeatureBlocks(
      *_config, _temporal_relationships, _vectors_map, *_context,
      should_update_history);

  blocks.insert(blocks.end(), temporal_feature_blocks.begin(),
                temporal_feature_blocks.end());
  return blocks;
}

std::vector<std::string_view> UDTDatasetFactory::toVectorOfStringViews(
    const MapInput& input) {
  verifyColumnNumberMapIsInitialized();
  std::vector<std::string_view> string_view_input(
      _column_number_map->numCols());
  for (const auto& [col_name, val] : input) {
    string_view_input[_column_number_map->at(col_name)] =
        std::string_view(val.data(), val.length());
  }
  return string_view_input;
}

std::string UDTDatasetFactory::concatenateWithDelimiter(
    const std::vector<std::string_view>& substrings, char delimiter) {
  if (substrings.empty()) {
    return "";
  }
  std::stringstream s;
  s << substrings[0];
  std::for_each(
      substrings.begin() + 1, substrings.end(),
      [&](const std::string_view& substr) { s << delimiter << substr; });
  return s.str();
}

}  // namespace thirdai::automl::data

CEREAL_REGISTER_TYPE(thirdai::automl::data::UDTDatasetFactory)