#pragma once
#include <cereal/access.hpp>
#include <auto_ml/src/dataset_factories/udt/DataTypes.h>
#include <auto_ml/src/dataset_factories/udt/FeatureComposer.h>
#include <dataset/src/DataSource.h>
#include <dataset/src/dataset_loaders/DatasetLoader.h>
#include <dataset/src/featurizers/TabularFeaturizer.h>
#include <cstdint>
#include <string>
#include <unordered_map>

namespace thirdai::automl::data {

class CategoricalMetadata {
 public:
  CategoricalMetadata() {}

  CategoricalMetadata(const ColumnDataTypes& data_types,
                      uint32_t text_pairgram_word_limit,
                      bool contextual_columns, uint32_t hash_range);

  void updateMetadata(const std::string& col_name, const MapInput& update);

  void updateMetadataBatch(const std::string& col_name,
                           const MapInputBatch& updates);

  const auto& metadataVectors() const { return _metadata_vectors; }

 private:
  static dataset::PreprocessedVectorsPtr fromDatasets(
      dataset::BoltDataset& features, dataset::BoltDataset& key_ids,
      dataset::ThreadSafeVocabulary& key_vocab, uint32_t feature_dim);

  void verifyMetadataExists(const std::string& col_name) {
    if (!_featurizers.count(col_name) || !_metadata_vectors.count(col_name) ||
        !_keys.count(col_name)) {
      throw std::invalid_argument("'" + col_name + "' is an invalid column.");
    }
  }

  uint32_t _text_pairgram_word_limit;
  bool _contextual_columns;
  uint32_t _hash_range;
  std::unordered_map<std::string, std::string> _keys;
  std::unordered_map<std::string, dataset::TabularFeaturizerPtr> _featurizers;
  PreprocessedVectorsMap _metadata_vectors;

  friend class cereal::access;
  template <class Archive>
  void serialize(Archive& archive) {
    archive(_text_pairgram_word_limit, _contextual_columns, _hash_range, _keys,
            _featurizers, _metadata_vectors);
  }
};

}  // namespace thirdai::automl::data