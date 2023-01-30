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
/**
 * Manages categorical metadata for UDT.
 *
 * Some datasets have categorical features that map to other static features.
 * For example, each sample in a user transactions dataset has a categorical
 * user ID feature. In turn, these user IDs may be associated with unchanging or
 * rarely changing ("static") features such as date and place of birth, country
 * of residence, and so on. Instead of being included in the transaction
 * records, the owner of the dataset may choose to store these features in a
 * separate "metadata" file that maps user IDs to their static features.
 *
 * This class loads feature vectors from metadata files. It then creates and
 * maintains maps between categorical features (e.g. user ID) and their metadata
 * vectors (e.g. vectors that encode date and place of birth).These vectors can
 * then be injected when UDT loads the main dataset (e.g. the transaction
 * dataset in the above example), thus providing additional features without
 * bloating up the main dataset.
 */
class CategoricalMetadata {
 public:
  CategoricalMetadata() {}

  /**
   * Looks through `data_types` for categorical data types that have a metadata
   * config object. It then loads vectors according to these metadata config
   * objects and creates maps between categorical features and their metadata
   * vectors.
   */
  CategoricalMetadata(const ColumnDataTypes& data_types,
                      uint32_t text_pairgram_word_limit,
                      bool contextual_columns, uint32_t hash_range);

  /**
   * Updates metadata for a categorical column (e.g. user ID) with a single new
   * metadata sample.
   */
  void updateMetadata(const std::string& col_name, const MapInput& update);

  /**
   * Updates metadata for a categorical column (e.g. user ID) with a batch of
   * new metadata samples.
   */
  void updateMetadataBatch(const std::string& col_name,
                           const MapInputBatch& updates);

  /**
   * Returns a map from categorical column names to their metadata vector maps.
   */
  const auto& metadataVectors() const { return _metadata_vectors; }

 private:
  static dataset::PreprocessedVectorsPtr metadataVectorMap(
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