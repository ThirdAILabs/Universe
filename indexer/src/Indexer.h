#pragma once

#include <hashing/src/DWTA.h>
#include <hashing/src/DensifiedMinHash.h>
#include <hashing/src/FastSRP.h>
#include <hashing/src/HashFunction.h>
#include <hashing/src/SRP.h>
#include <dataset/src/DataLoader.h>
#include <dataset/src/Datasets.h>
#include <dataset/src/StreamingGenericDatasetLoader.h>
#include <dataset/src/blocks/BlockInterface.h>
#include <dataset/src/blocks/Text.h>
#include <dataset/src/utils/TextEncodingUtils.h>
#include <exceptions/src/Exceptions.h>
#include <indexer/src/flash.h>
#include <spdlog/fmt/bundled/core.h>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>

namespace thirdai::automl::deployment {

class FlashIndexConfig {
 public:
  explicit FlashIndexConfig(const std::string& config_file_path)
      : _config_file_path(config_file_path),
        _hash_function(),
        _input_dim(100000),
        _num_tables(0),
        _range(0),
        _hashes_per_table(0) {}

  void setParametersFromConfig() {}

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

  static void save(const std::string& file_name) { (void)file_name; }

  std::optional<std::string> _config_file_path;
  std::string _hash_function;
  std::optional<uint32_t> _input_dim;
  uint32_t _num_tables;
  uint32_t _range;
  uint32_t _hashes_per_table;
};

using FlashIndexConfigPtr = std::shared_ptr<FlashIndexConfig>;

class Indexer : public std::enable_shared_from_this<Indexer> {
 public:
  explicit Indexer(const std::string& config_file_path)
      : _flash_index_config(config_file_path),
        _dimension_for_encodings(
            dataset::TextEncodingUtils::DEFAULT_TEXT_ENCODING_DIM),
        _batch_size(32) {}

  /**
   * @brief Builds a Flash index
   *
   * @param file_name
   * @return std::shared_ptr<Indexer>
   */
  template <typename LABEL_T>
  std::shared_ptr<Indexer> buildFlashIndex(const std::string& file_name) {
    auto data = loadDataInMemory(file_name);
    _flash_index = Flash<LABEL_T>(*_flash_index_config.getHashFunction());
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

  FlashIndexConfig _flash_index_config;

  std::unique_ptr<Flash<uint32_t>> _flash_index;
  uint32_t _dimension_for_encodings;
  uint32_t _batch_size;

  // uint32_t batch_size;
};

}  // namespace thirdai::automl::deployment