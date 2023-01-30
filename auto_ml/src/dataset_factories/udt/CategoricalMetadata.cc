#include "CategoricalMetadata.h"

namespace thirdai::automl::data {

CategoricalMetadata::CategoricalMetadata(ColumnDataTypes data_types,
                                         uint32_t text_pairgram_word_limit,
                                         bool contextual_columns,
                                         uint32_t hash_range)
    : _data_types(std::move(data_types)),
      _text_pairgram_word_limit(text_pairgram_word_limit),
      _contextual_columns(contextual_columns),
      _hash_range(hash_range) {
  for (const auto& [name, type] : _data_types) {
    if (auto categorical = asCategorical(type)) {
      if (categorical->metadata_config) {
        _metadata_vectors[name] = loadVectors(name);
      }
    }
  }
}

void CategoricalMetadata::updateMetadata(const std::string& col_name,
                                         const MapInput& update) {
  verifyMetadataExists(col_name);
  auto metadata_config =
      asCategorical(_data_types.at(col_name))->metadata_config;

  dataset::MapSampleRef update_ref(update);
  auto vec = _featurizers.at(col_name)->makeInputVector(update_ref);

  const auto& key = update.at(metadata_config->key);
  _metadata_vectors.at(col_name)->vectors[key] = vec;
}

void CategoricalMetadata::updateMetadataBatch(const std::string& col_name,
                                              const MapInputBatch& updates) {
  verifyMetadataExists(col_name);
  auto metadata_config =
      asCategorical(_data_types.at(col_name))->metadata_config;

  dataset::MapBatchRef updates_ref(updates);
  std::vector<BoltVector> batch =
      _featurizers.at(col_name)->featurize(updates_ref).at(0);

  for (uint32_t update_idx = 0; update_idx < updates.size(); update_idx++) {
    const auto& key = updates.at(update_idx).at(metadata_config->key);
    _metadata_vectors.at(col_name)->vectors[key] = batch.at(update_idx);
  }
}

dataset::PreprocessedVectorsPtr CategoricalMetadata::loadVectors(
    const std::string& name) {
  auto target = asCategorical(_data_types.at(name));

  if (!target->metadata_config) {
    throw std::invalid_argument("The given categorical column (" + name +
                                ") does not have a metadata config.");
  }

  auto metadata = target->metadata_config;

  // TODO(Geordie): Support other data sources
  auto data_source = dataset::FileDataSource::make(metadata->metadata_file);

  auto input_blocks = FeatureComposer::makeNonTemporalFeatureBlocks(
      /* data_types= */ metadata->column_data_types,
      /* target= */ metadata->key,
      /* temporal_relationships= */ {},
      /* vectors_map= */ {},
      /* text_pairgrams_word_limit= */ _text_pairgram_word_limit,
      /* contextual_columns= */ _contextual_columns);

  auto key_vocab = dataset::ThreadSafeVocabulary::make(
      /* vocab_size= */ 0, /* limit_vocab_size= */ false);
  auto label_block =
      dataset::StringLookupCategoricalBlock::make(metadata->key, key_vocab);

  _featurizers[name] = dataset::TabularFeaturizer::make(
      /* input_blocks= */ std::move(input_blocks),
      /* label_blocks= */ {std::move(label_block)},
      /* has_header= */ true, /* delimiter= */ metadata->delimiter,
      /* parallel= */ true, /* hash_range= */ _hash_range);

  // Here we set parallel=true because there are no temporal
  // relationships in the metadata file.
  dataset::DatasetLoader loader(
      /* source= */ data_source,
      /* processor= */ _featurizers[name],
      /* shuffle = */ false);

  return preprocessedVectorsFromDataset(loader, *key_vocab);
}

dataset::PreprocessedVectorsPtr
CategoricalMetadata::preprocessedVectorsFromDataset(
    dataset::DatasetLoader& dataset_loader,
    dataset::ThreadSafeVocabulary& key_vocab) {
  // The batch size does not really matter here because we are storing these
  // vectors as metadata, not training on them. Thus, we choose the somewhat
  // arbitrary value 2048 since it is large enough to use all threads.
  auto [datasets, ids] = dataset_loader.loadAll(/* batch_size = */ 2048);

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

}  // namespace thirdai::automl::data