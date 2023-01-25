#include "UDTDatasetFactory.h"
#include <cereal/archives/binary.hpp>
#include <dataset/src/DataSource.h>
#include <dataset/src/blocks/BlockInterface.h>
#include <dataset/src/blocks/InputTypes.h>
#include <stdexcept>

namespace thirdai::automl::data {

UDTDatasetFactory::UDTDatasetFactory(
    const UDTConfigPtr& config, bool force_parallel,
    uint32_t text_pairgram_word_limit, bool contextual_columns,
    std::optional<dataset::RegressionBinningStrategy> regression_binning)
    : _temporal_relationships(TemporalRelationshipsAutotuner::autotune(
          config->data_types, config->provided_relationships,
          config->lookahead)),
      _config(FeatureComposer::verifyConfigIsValid(config,
                                                   _temporal_relationships)),
      _context(std::make_shared<TemporalContext>()),
      _parallel(_temporal_relationships.empty() || force_parallel),
      _text_pairgram_word_limit(text_pairgram_word_limit),
      _contextual_columns(contextual_columns),
      _normalize_target_categories(false),
      _regression_binning(regression_binning),
      _vectors_map(processAllMetadata()),
      _labeled_history_updating_processor(makeLabeledUpdatingProcessor()),
      _unlabeled_non_updating_processor(makeUnlabeledNonUpdatingProcessor()) {}

dataset::DatasetLoaderPtr UDTDatasetFactory::getLabeledDatasetLoader(
    dataset::DataSourcePtr data_source, bool training) {
  auto column_number_map =
      makeColumnNumberMapFromHeader(*data_source, _config->delimiter);
  _column_number_to_name = column_number_map.getColumnNumToColNameMap();

  // The featurizer will treat the next line as a header
  // Restart so featurizer does not skip a sample.
  data_source->restart();

  _labeled_history_updating_processor->updateColumnNumbers(column_number_map);
  _unlabeled_non_updating_processor->updateColumnNumbers(column_number_map);

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

  auto data_source =
      dataset::SimpleFileDataSource::make(metadata->metadata_file);

  auto column_numbers =
      makeColumnNumberMapFromHeader(*data_source, metadata->delimiter);
  data_source->restart();

  auto input_blocks = buildMetadataInputBlocks(*metadata);

  auto key_vocab = dataset::ThreadSafeVocabulary::make(
      /* vocab_size= */ 0, /* limit_vocab_size= */ false);
  auto label_block =
      dataset::StringLookupCategoricalBlock::make(metadata->key, key_vocab);

  _metadata_processors[col_name] = dataset::TabularFeaturizer::make(
      /* input_blocks= */ std::move(input_blocks),
      /* label_blocks= */ {std::move(label_block)},
      /* has_header= */ true, /* delimiter= */ metadata->delimiter,
      /* parallel= */ true, /* hash_range= */ _config->hash_range);

  _metadata_processors[col_name]->updateColumnNumbers(column_numbers);

  // Here we set parallel=true because there are no temporal
  // relationships in the metadata file.
  dataset::DatasetLoader metadata_source(
      /* source= */ data_source,
      /* processor= */ _metadata_processors[col_name],
      /* shuffle = */ false);

  return preprocessedVectorsFromDataset(metadata_source, *key_vocab);
}

ColumnNumberMap UDTDatasetFactory::makeColumnNumberMapFromHeader(
    dataset::DataSource& data_source, char delimiter) {
  auto header = data_source.nextLine();
  if (!header) {
    throw std::invalid_argument(
        "The dataset must have a header that contains column names.");
  }

  return {*header, delimiter};
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
    dataset::DatasetLoader& dataset_loader,
    dataset::ThreadSafeVocabulary& key_vocab) {
  // The batch size does not really matter here because we are storing these
  // vectors as metadata, not training on them. Thus, we choose the somewhat
  // arbitrary value 2048 since it is large enough to use all threads.
  auto [datasets, ids] = dataset_loader.loadInMemory(/* batch_size = */ 2048);

  if (datasets.size() != 1) {
    throw std::runtime_error(
        "For now, the featurizer should return just a single input "
        "dataset.");
  }
  auto vectors = datasets.at(0);

  std::unordered_map<std::string, BoltVector> preprocessed_vectors(ids->len());

  for (uint32_t batch = 0; batch < vectors->numBatches(); batch++) {
    for (uint32_t vec = 0; vec < vectors->at(batch).getBatchSize(); vec++) {
      auto id = ids->at(batch)[vec].active_neurons[0];
      auto key = key_vocab.getString(id);
      preprocessed_vectors[key] = std::move(vectors->at(batch)[vec]);
    }
  }

  return std::make_shared<dataset::PreprocessedVectors>(
      std::move(preprocessed_vectors), dataset_loader.getInputDim());
}

void UDTDatasetFactory::updateMetadata(const std::string& col_name,
                                       const MapInput& update) {
  verifyColumnMetadataExists(col_name);

  auto metadata_config = getColumnMetadataConfig(col_name);

  dataset::MapSampleRef update_ref(update);
  auto vec = _metadata_processors.at(col_name)->makeInputVector(update_ref);

  const auto& key = update.at(metadata_config->key);
  _vectors_map.at(col_name)->vectors[key] = vec;
}

void UDTDatasetFactory::updateMetadataBatch(const std::string& col_name,
                                            const MapInputBatch& updates) {
  verifyColumnMetadataExists(col_name);
  auto metadata_config = getColumnMetadataConfig(col_name);

  dataset::MapBatchRef updates_ref(updates);
  auto batch =
      _metadata_processors.at(col_name)->createBatch(updates_ref).at(0);

  for (uint32_t update_idx = 0; update_idx < updates.size(); update_idx++) {
    const auto& key = updates.at(update_idx).at(metadata_config->key);
    _vectors_map.at(col_name)->vectors[key] = batch[update_idx];
  }
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