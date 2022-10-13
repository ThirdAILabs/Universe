#pragma once

#include <cereal/access.hpp>
#include <cereal/archives/portable_binary.hpp>
#include <cereal/types/memory.hpp>
#include <hashing/src/DWTA.h>
#include <hashing/src/DensifiedMinHash.h>
#include <hashing/src/FastSRP.h>
#include <_types/_uint32_t.h>
#include <dataset/src/DataLoader.h>
#include <dataset/src/Datasets.h>
#include <dataset/src/StreamingGenericDatasetLoader.h>
#include <dataset/src/blocks/BlockInterface.h>
#include <dataset/src/blocks/Text.h>
#include <dataset/src/utils/TextEncodingUtils.h>
#include <exceptions/src/Exceptions.h>
#include <indexer/src/Flash.h>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>

namespace thirdai::automl::deployment {

class FlashIndexConfig {
 public:
  FlashIndexConfig(std::string hash_function, uint32_t input_dim,
                   uint32_t batch_size, uint32_t num_tables,
                   uint32_t hashes_per_table, uint32_t range)
      : _hash_function(std::move(hash_function)),
        _input_dim(input_dim),
        _batch_size(batch_size),
        _num_tables(num_tables),
        _range(range),
        _hashes_per_table(hashes_per_table) {}

  // FlashIndexConfig(FlashIndexConfig&&) = default;

  void saveFlashIndexConfig(const std::string& config_file_name) {
    std::stringstream output;
    cereal::PortableBinaryInputArchive output_archive(output);

    output_archive(*this);

    std::ofstream filestream =
        dataset::SafeFileIO::ofstream(config_file_name, std::ios::binary);

    std::string string_representation = output.str();
    filestream.write(string_representation.data(),
                     string_representation.size());
  }

  static std::shared_ptr<FlashIndexConfig> loadAndSetParametersFromConfig(
      const std::string& config_file_name) {
    std::ifstream filestream =
        dataset::SafeFileIO::ifstream(config_file_name, std::ios::binary);

    std::stringstream buffer;
    buffer << filestream.rdbuf();

    cereal::PortableBinaryInputArchive input_archive(buffer);
    std::shared_ptr<FlashIndexConfig> deserialized_config(
        new FlashIndexConfig());
    input_archive(*deserialized_config);

    return deserialized_config;
  }

  std::shared_ptr<hashing::HashFunction> getHashFunction() const {
    if (_hash_function == "DensiFiedMinHash") {
      return std::make_shared<hashing::DensifiedMinHash>(
          /* hashes_per_table = */ _hashes_per_table,
          /* num_tables = */ _num_tables, /* range = */ _range);
    }
    if (_hash_function == "DWTA") {
      return std::make_shared<hashing::DWTAHashFunction>(
          /* input_dim = */ *_input_dim,
          /* hashes_per_table = */ _hashes_per_table,
          /* num_tables = */ _num_tables, /* range_pow = */ _range);
    }
    if (_hash_function == "FastSRP") {
      return std::make_shared<hashing::FastSRP>(
          /* input_dim = */ *_input_dim,
          /* hashes_per_table = */ _hashes_per_table,
          /* num_tables = */ _num_tables);
    }
    throw exceptions::NotImplemented("Unsupported Hash Function");
  }

 private:
  // Private constructor for cereal
  FlashIndexConfig() {}

  friend class cereal::access;
  template <typename Archive>
  void serialize(Archive& archive) {
    archive(_hash_function, _input_dim, _batch_size, _num_tables, _range,
            _hashes_per_table);
  }

  std::string _hash_function;
  std::optional<uint32_t> _input_dim;
  uint32_t _batch_size;
  uint32_t _num_tables;
  uint32_t _range;
  uint32_t _hashes_per_table;
};

using FlashIndexConfigPtr = std::shared_ptr<FlashIndexConfig>;

class Indexer : public std::enable_shared_from_this<Indexer> {
 public:
  explicit Indexer(FlashIndexConfigPtr flash_index_config)
      : _flash_index_config(std::move(flash_index_config)),
        _dimension_for_encodings(
            dataset::TextEncodingUtils::DEFAULT_TEXT_ENCODING_DIM) {}

  explicit Indexer(const std::string& config_file_path)
      : _dimension_for_encodings(
            dataset::TextEncodingUtils::DEFAULT_TEXT_ENCODING_DIM),
        _batch_size(32) {
    (void)config_file_path;
  }

  static auto make(const FlashIndexConfigPtr& flash_index_config) {
    return Indexer(flash_index_config);
  }

  static std::shared_ptr<Indexer> buildIndexerFromSerializedConfig(
      const std::string& config_file_name) {
    auto flash_index_config =
        FlashIndexConfig::loadAndSetParametersFromConfig(config_file_name);

    return std::make_shared<Indexer>(flash_index_config);
  }

  /**
   * @brief Builds a Flash index
   *
   * @param file_name
   * @return std::shared_ptr<Indexer>
   */
  template <typename LABEL_T>
  std::shared_ptr<Indexer> buildFlashIndex(const std::string& file_name) {
    auto data = loadDataInMemory(file_name);
    _flash_index = std::make_unique<Flash<LABEL_T>>(
        *_flash_index_config->getHashFunction());
    _flash_index->addDataset(*data);

    return shared_from_this();
  }

  template <typename LABEL_T>
  std::vector<std::vector<LABEL_T>> queryIndex(const std::string& query_file) {
    auto query_data = loadDataInMemory(query_file);
    std::vector<std::vector<LABEL_T>> results = _flash_index->queryBatch(
        /* batch = */ query_data, /* top_k = */ 5, /* pad_zeros = */ true);
    return results;
  }

  /**
   * @brief Builds a flash index by reading from a CSV file containing pairs of
   * incorrect and correct queries.
   * The first column is expected to contain incorrect queries and the second
   * one to contain the corresponding correct queries. The function tokenizes
   * the incorrect query and builds the index using the correct queries instead.
   *
   * @param file_name
   * @return std::shared_ptr<Indexer>
   */
  std::shared_ptr<Indexer> buildFlashIndexPair(const std::string& file_name) {
    (void)file_name;
    return shared_from_this();
  }

 private:
  dataset::BoltDatasetPtr loadDataInMemory(const std::string& file_name) const {
    dataset::TextBlockPtr char_trigram_block =
        dataset::CharKGramTextBlock::make(
            /* col = */ 0, /* k = */ 3,
            /* dim = */ _dimension_for_encodings);

    dataset::TextBlockPtr char_four_gram = dataset::CharKGramTextBlock::make(
        /* col = */ 0, /* k = */ 4,
        /* dim = */ _dimension_for_encodings);

    std::vector<std::shared_ptr<dataset::Block>> input_blocks{
        char_trigram_block, char_four_gram};

    auto data_loader = dataset::StreamingGenericDatasetLoader(
        /* filename = */ file_name, /* input_blocks = */ input_blocks,
        /* label_blocks = */ {}, /* batch_size = */ _batch_size);

    auto [data, _] = data_loader.loadInMemory();

    return data;
  }

  std::shared_ptr<FlashIndexConfig> _flash_index_config;

  std::unique_ptr<Flash<uint32_t>> _flash_index;
  uint32_t _dimension_for_encodings;
  uint32_t _batch_size;

  // uint32_t batch_size;
};

using IndexerPtr = std::shared_ptr<Indexer>;

}  // namespace thirdai::automl::deployment