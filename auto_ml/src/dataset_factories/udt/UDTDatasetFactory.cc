#include "UDTDatasetFactory.h"
#include <cereal/archives/binary.hpp>
#include <auto_ml/src/dataset_factories/udt/ColumnNumberMap.h>
#include <dataset/src/DataSource.h>
#include <stdexcept>

namespace thirdai::automl::data {

dataset::DatasetLoaderPtr UDTDatasetFactory::getLabeledDatasetLoader(
    dataset::DataSourcePtr data_source, bool training) {
  auto current_column_number_map =
      makeColumnNumberMap(*data_source, _config->delimiter);

  if (!_column_number_map) {
    _column_number_map = std::move(current_column_number_map);
    _column_number_to_name = _column_number_map->getColumnNumToColNameMap();
  } else if (!_column_number_map->equals(*current_column_number_map)) {
    throw std::invalid_argument("Column positions should not change.");
  }

  if (!_labeled_history_updating_processor) {
    _labeled_history_updating_processor =
        makeLabeledUpdatingProcessor(*_column_number_map);
  }

  // We initialize the inference batch processor here because we need the
  // column number map.
  if (!_unlabeled_non_updating_processor) {
    _unlabeled_non_updating_processor =
        makeUnlabeledNonUpdatingProcessor(*_column_number_map);
  }

  // The batch processor will treat the next line as a header
  // Restart so batch processor does not skip a sample.
  data_source->restart();

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
      dataset::SimpleFileDataSource::make(metadata->metadata_file,
                                          /* target_batch_size= */ 2048);

  _metadata_column_number_maps[col_name] =
      makeColumnNumberMap(*data_source, metadata->delimiter);

  auto input_blocks = buildMetadataInputBlocks(
      *metadata, *_metadata_column_number_maps[col_name]);

  auto key_vocab = dataset::ThreadSafeVocabulary::make(
      /* vocab_size= */ 0, /* limit_vocab_size= */ false);
  auto label_block = dataset::StringLookupCategoricalBlock::make(
      _metadata_column_number_maps[col_name]->at(metadata->key), key_vocab);

  _metadata_processors[col_name] = dataset::GenericBatchProcessor::make(
      /* input_blocks= */ std::move(input_blocks),
      /* label_blocks= */ {std::move(label_block)},
      /* has_header= */ false, /* delimiter= */ metadata->delimiter,
      /* parallel= */ true, /* hash_range= */ _config->hash_range);

  // Here we set parallel=true because there are no temporal
  // relationships in the metadata file.
  dataset::DatasetLoader metadata_source(
      /* source= */ data_source,
      /* processor= */ _metadata_processors[col_name],
      /* shuffle = */ false);

  return preprocessedVectorsFromDataset(metadata_source, *key_vocab);
}

ColumnNumberMapPtr UDTDatasetFactory::makeColumnNumberMap(
    dataset::DataSource& data_source, char delimiter) {
  auto header = data_source.nextLine();
  if (!header) {
    throw std::invalid_argument(
        "The dataset must have a header that contains column names.");
  }

  return std::make_shared<ColumnNumberMap>(*header, delimiter);
}

std::vector<dataset::BlockPtr> UDTDatasetFactory::buildMetadataInputBlocks(
    const CategoricalMetadataConfig& metadata_config,
    const ColumnNumberMap& column_numbers) const {
  UDTConfig feature_config(
      /* data_types= */ metadata_config.column_data_types,
      /* temporal_tracking_relationships= */ {},
      /* target= */ metadata_config.key,
      /* n_target_classes= */ 0);
  TemporalRelationships empty_temporal_relationships;

  PreprocessedVectorsMap empty_vectors_map;

  return FeatureComposer::makeNonTemporalFeatureBlocks(
      feature_config, empty_temporal_relationships, column_numbers,
      empty_vectors_map, _text_pairgram_word_limit, _contextual_columns);
}

dataset::PreprocessedVectorsPtr
UDTDatasetFactory::preprocessedVectorsFromDataset(
    dataset::DatasetLoader& dataset_loader,
    dataset::ThreadSafeVocabulary& key_vocab) {
  auto [datasets, ids] = dataset_loader.loadInMemory();

  if (datasets.size() != 1) {
    throw std::runtime_error(
        "For now, the batch processor should return just a single input "
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

  auto vec = boltVectorFromInput(*_metadata_processors.at(col_name),
                                 *_metadata_column_number_maps.at(col_name),
                                 metadata_config->delimiter, update);

  const auto& key = update.at(metadata_config->key);
  _vectors_map.at(col_name)->vectors[key] = vec;
}

void UDTDatasetFactory::updateMetadataBatch(const std::string& col_name,
                                            const MapInputBatch& updates) {
  verifyColumnMetadataExists(col_name);
  auto metadata_config = getColumnMetadataConfig(col_name);

  auto batch = _metadata_processors.at(col_name)
                   ->createBatch(lineInputBatchFromMapInputBatch(
                       *_metadata_column_number_maps.at(col_name),
                       metadata_config->delimiter, updates))
                   .at(0);

  for (uint32_t update_idx = 0; update_idx < updates.size(); update_idx++) {
    const auto& key = updates.at(update_idx).at(metadata_config->key);
    _vectors_map.at(col_name)->vectors[key] = batch[update_idx];
  }
}

dataset::GenericBatchProcessorPtr
UDTDatasetFactory::makeLabeledUpdatingProcessor(
    const ColumnNumberMap& column_number_map) {
  if (!_config->data_types.count(_config->target)) {
    throw std::invalid_argument(
        "data_types parameter must include the target column.");
  }

  dataset::BlockPtr label_block = getLabelBlock(column_number_map);

  auto input_blocks = buildInputBlocks(/* column_numbers= */ column_number_map,
                                       /* should_update_history= */ true);

  auto processor = dataset::GenericBatchProcessor::make(
      std::move(input_blocks), {label_block}, /* has_header= */ true,
      /* delimiter= */ _config->delimiter, /* parallel= */ _parallel,
      /* hash_range= */ _config->hash_range);
  return processor;
}

dataset::BlockPtr UDTDatasetFactory::getLabelBlock(
    const ColumnNumberMap& column_number_map) {
  auto target_type = _config->data_types.at(_config->target);
  auto target_col_num = column_number_map.at(_config->target);

  if (asCategorical(target_type)) {
    auto target_config = asCategorical(target_type);
    if (!_config->n_target_classes) {
      throw std::invalid_argument(
          "n_target_classes must be specified for a classification task.");
    }
    if (_config->integer_target) {
      return dataset::NumericalCategoricalBlock::make(
          /* col= */ target_col_num,
          /* n_classes= */ _config->n_target_classes.value(),
          /* delimiter= */ target_config->delimiter,
          /* normalize_categories= */ _normalize_target_categories);
    }
    if (!_vocabs.count(_config->target)) {
      _vocabs[_config->target] = dataset::ThreadSafeVocabulary::make(
          /* vocab_size= */ _config->n_target_classes.value());
    }
    return dataset::StringLookupCategoricalBlock::make(
        /* col= */ target_col_num, /* vocab= */ _vocabs.at(_config->target),
        /* delimiter= */ target_config->delimiter,
        /* normalize_categories= */ _normalize_target_categories);
  }
  if (asNumerical(target_type)) {
    if (!_regression_binning) {
      throw std::logic_error(
          "Regression binning must be set for numerical outputs.");
    }

    return dataset::RegressionCategoricalBlock::make(
        /* col= */ target_col_num,
        /* binning_strategy= */ _regression_binning.value(),
        /* correct_label_radius= */
        UDTConfig::REGRESSION_CORRECT_LABEL_RADIUS,
        /* labels_sum_to_one= */ true);
  }
  throw std::invalid_argument(
      "Target column must have type numerical or categorical.");
}

std::vector<dataset::BlockPtr> UDTDatasetFactory::buildInputBlocks(
    const ColumnNumberMap& column_numbers, bool should_update_history) {
  std::vector<dataset::BlockPtr> blocks =
      FeatureComposer::makeNonTemporalFeatureBlocks(
          *_config, _temporal_relationships, column_numbers, _vectors_map,
          _text_pairgram_word_limit, _contextual_columns);

  if (_temporal_relationships.empty()) {
    return blocks;
  }

  auto temporal_feature_blocks = FeatureComposer::makeTemporalFeatureBlocks(
      *_config, _temporal_relationships, column_numbers, _vectors_map,
      *_context, should_update_history);

  blocks.insert(blocks.end(), temporal_feature_blocks.begin(),
                temporal_feature_blocks.end());
  return blocks;
}

std::vector<std::string_view> UDTDatasetFactory::toVectorOfStringViews(
    const ColumnNumberMap& column_number_map, char delimiter,
    const MapInput& input) {
  (void)delimiter;
  std::vector<std::string_view> string_view_input(column_number_map.numCols());
  for (const auto& [col_name, val] : input) {
    string_view_input[column_number_map.at(col_name)] =
        std::string_view(val.data(), val.length());
  }
  return string_view_input;
}

std::vector<std::string> UDTDatasetFactory::lineInputBatchFromMapInputBatch(
    const ColumnNumberMap& column_number_map, char delimiter,
    const MapInputBatch& input_maps) {
  std::vector<std::string> string_batch(input_maps.size());
  for (uint32_t i = 0; i < input_maps.size(); i++) {
    auto vals =
        toVectorOfStringViews(column_number_map, delimiter, input_maps[i]);
    string_batch[i] = concatenateWithDelimiter(vals, delimiter);
  }
  return string_batch;
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