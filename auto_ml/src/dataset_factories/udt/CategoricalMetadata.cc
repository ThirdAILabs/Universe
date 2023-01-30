#include "CategoricalMetadata.h"
#include <dataset/src/Datasets.h>
#include <dataset/src/utils/ThreadSafeVocabulary.h>

namespace thirdai::automl::data {

CategoricalMetadata::CategoricalMetadata(const ColumnDataTypes& data_types,
                                         uint32_t text_pairgram_word_limit,
                                         bool contextual_columns,
                                         uint32_t hash_range)
    : _text_pairgram_word_limit(text_pairgram_word_limit),
      _contextual_columns(contextual_columns),
      _hash_range(hash_range) {
  for (const auto& [name, type] : data_types) {
    if (auto categorical = asCategorical(type)) {
      if (auto metadata_config = categorical->metadata_config) {
        auto input_blocks = FeatureComposer::makeNonTemporalFeatureBlocks(
            /* data_types= */ metadata_config->column_data_types,
            /* target= */ metadata_config->key,
            /* temporal_relationships= */ {},
            /* vectors_map= */ {},
            /* text_pairgrams_word_limit= */ _text_pairgram_word_limit,
            /* contextual_columns= */ _contextual_columns);

        auto key_vocab = dataset::ThreadSafeVocabulary::make(
            /* vocab_size= */ 0, /* limit_vocab_size= */ false);
        auto label_block = dataset::StringLookupCategoricalBlock::make(
            metadata_config->key, key_vocab);

        auto featurizer = dataset::TabularFeaturizer::make(
            /* input_blocks= */ std::move(input_blocks),
            /* label_blocks= */ {std::move(label_block)},
            /* has_header= */ true, /* delimiter= */ metadata_config->delimiter,
            /* parallel= */ true, /* hash_range= */ _hash_range);

        auto data_source =
            dataset::FileDataSource::make(metadata_config->metadata_file);

        // Here we set parallel=true because there are no temporal
        // relationships in the metadata file.
        dataset::DatasetLoader loader(
            /* source= */ data_source,
            /* processor= */ featurizer,
            /* shuffle = */ false);

        // The batch size does not really matter here because we are storing
        // these vectors as metadata, not training on them. Thus, we choose the
        // somewhat arbitrary value 2048 since it is large enough to use all
        // threads.
        auto [features, key_ids] = loader.loadAll(/* batch_size = */ 2048);
        assert(features.size() == 1);

        _keys[name] = metadata_config->key;
        _featurizers[name] = featurizer;
        _metadata_vectors[name] = metadataVectorMap(
            /* features= */ *features.at(0), /* key_ids= */ *key_ids,
            /* key_vocab= */ *key_vocab,
            /* feature_dim= */ loader.getInputDim());
      }
    }
  }
}

void CategoricalMetadata::updateMetadata(const std::string& col_name,
                                         const MapInput& update) {
  verifyMetadataExists(col_name);

  dataset::MapSampleRef update_ref(update);
  auto vec = _featurizers.at(col_name)->makeInputVector(update_ref);

  const auto& key = update.at(_keys.at(col_name));
  _metadata_vectors.at(col_name)->vectors[key] = vec;
}

void CategoricalMetadata::updateMetadataBatch(const std::string& col_name,
                                              const MapInputBatch& updates) {
  verifyMetadataExists(col_name);

  dataset::MapBatchRef updates_ref(updates);
  std::vector<BoltVector> batch =
      _featurizers.at(col_name)->featurize(updates_ref).at(0);

  for (uint32_t update_idx = 0; update_idx < updates.size(); update_idx++) {
    const auto& key = updates.at(update_idx).at(_keys.at(col_name));
    _metadata_vectors.at(col_name)->vectors[key] = batch.at(update_idx);
  }
}

dataset::PreprocessedVectorsPtr CategoricalMetadata::metadataVectorMap(
    dataset::BoltDataset& features, dataset::BoltDataset& key_ids,
    dataset::ThreadSafeVocabulary& key_vocab, uint32_t feature_dim) {
  std::unordered_map<std::string, BoltVector> preprocessed_vectors(
      key_ids.len());

  for (uint32_t batch = 0; batch < features.numBatches(); batch++) {
    for (uint32_t vec = 0; vec < features.at(batch).getBatchSize(); vec++) {
      auto key_id = key_ids.at(batch)[vec].active_neurons[0];
      auto key = key_vocab.getString(key_id);
      preprocessed_vectors[key] = std::move(features.at(batch)[vec]);
    }
  }

  return std::make_shared<dataset::PreprocessedVectors>(
      std::move(preprocessed_vectors), feature_dim);
}

}  // namespace thirdai::automl::data