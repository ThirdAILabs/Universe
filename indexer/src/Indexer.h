#pragma once

#include <cereal/access.hpp>
#include <cereal/archives/binary.hpp>
#include <cereal/types/base_class.hpp>
#include <cereal/types/memory.hpp>
#include <cereal/types/optional.hpp>
#include <hashing/src/DWTA.h>
#include <hashing/src/DensifiedMinHash.h>
#include <hashing/src/FastSRP.h>
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

namespace thirdai::bolt {

class IndexerConfig {
 public:
  IndexerConfig(std::string hash_function, uint32_t num_tables,
                uint32_t hashes_per_table, uint32_t input_dim,
                uint32_t batch_size = 100, uint32_t range = 1000000)
      : _hash_function(std::move(hash_function)),
        _num_tables(num_tables),
        _hashes_per_table(hashes_per_table),
        _input_dim(input_dim),
        _batch_size(batch_size),
        _range(range) {}

  // move and copy constructors
  IndexerConfig(IndexerConfig&& other) = default;
  IndexerConfig(const IndexerConfig& other) = default;

  // overloaded operator mainly for testing
  auto operator==(IndexerConfig* rhs) const {
    return this->_hash_function == rhs->_hash_function &&
           this->_num_tables == rhs->_num_tables &&
           this->_hashes_per_table == rhs->_hashes_per_table &&
           this->_input_dim == rhs->_input_dim &&
           this->_batch_size == rhs->_batch_size && this->_range == rhs->_range;
  }

  void save(const std::string& config_file_name) const {
    std::stringstream output;
    cereal::BinaryOutputArchive output_archive(output);

    output_archive(*this);

    std::ofstream filestream =
        dataset::SafeFileIO::ofstream(config_file_name, std::ios::binary);

    std::string string_representation = output.str();
    filestream.write(string_representation.data(),
                     string_representation.size());
  }

  static std::shared_ptr<IndexerConfig> load(
      const std::string& config_file_name) {
    std::ifstream filestream =
        dataset::SafeFileIO::ifstream(config_file_name, std::ios::binary);

    std::stringstream buffer;
    buffer << filestream.rdbuf();

    cereal::BinaryInputArchive input_archive(buffer);
    std::shared_ptr<IndexerConfig> deserialized_config(new IndexerConfig());
    input_archive(*deserialized_config);

    return deserialized_config;
  }

  std::shared_ptr<hashing::HashFunction> getHashFunction() const {
    if (_hash_function == "DensifiedMinHash") {
      return std::make_shared<hashing::DensifiedMinHash>(
          /* hashes_per_table = */ _hashes_per_table,
          /* num_tables = */ _num_tables, /* range = */ _range);
    }
    if (_hash_function == "DWTA") {
      return nullptr;
      //   return std::make_shared<hashing::DWTAHashFunction>(
      //       /* input_dim = */ _input_dim,
      //       /* hashes_per_table = */ _hashes_per_table,
      //       /* num_tables = */ _num_tables, /* range_pow = */ _range);
    }
    if (_hash_function == "FastSRP") {
      return nullptr;
      //   return std::make_shared<hashing::FastSRP>(
      //       /* input_dim = */ _input_dim,
      //       /* hashes_per_table = */ _hashes_per_table,
      //       /* num_tables = */ _num_tables);
    }
    throw exceptions::NotImplemented("Unsupported Hash Function");
  }

  constexpr uint32_t batch_size() const { return _batch_size; }

 private:
  std::string _hash_function;
  uint32_t _num_tables;
  uint32_t _hashes_per_table;

  std::optional<uint32_t> _input_dim;
  uint32_t _batch_size;
  uint32_t _range;

  // Private constructor for cereal
  IndexerConfig() {}

  friend class cereal::access;
  template <class Archive>
  void serialize(Archive& archive) {
    archive(_hash_function, _input_dim, _batch_size, _num_tables, _range,
            _hashes_per_table);
  }
};

using IndexerConfigPtr = std::shared_ptr<IndexerConfig>;

class Indexer : public std::enable_shared_from_this<Indexer> {
  const uint32_t TOP_K = 10;

 public:
  explicit Indexer(IndexerConfigPtr flash_index_config)
      : _flash_index_config(std::move(flash_index_config)),
        _dimension_for_encodings(
            dataset::TextEncodingUtils::DEFAULT_TEXT_ENCODING_DIM) {}

  static Indexer make(IndexerConfigPtr flash_index_config) {
    return Indexer(std::move(flash_index_config));
  }

  static Indexer buildIndexerFromSerializedConfig(
      const std::string& config_file_name) {
    auto flash_index_config = IndexerConfig::load(config_file_name);

    return Indexer::make(flash_index_config);
  }

  /**
   * @brief Builds a Flash index
   *
   * @param file_name
   * @return std::shared_ptr<Indexer>
   *
   * TODO(blaise): Combine buildFlashIndex and buildFlashIndexPair into one
   * function since they are pretty much indentical
   */
  std::shared_ptr<Indexer> buildFlashIndex(const std::string& file_name) {
    auto data = loadDataInMemory(file_name, 0);
    _flash_index = std::make_unique<Flash<uint32_t>>(
        _flash_index_config->getHashFunction());
    _flash_index->addDataset(*data);

    return shared_from_this();
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
    auto data = loadDataInMemory(file_name, 1);
    _flash_index = std::make_unique<Flash<uint32_t>>(
        _flash_index_config->getHashFunction());
    _flash_index->addDataset(*data);

    return shared_from_this();
  }

  std::vector<std::vector<std::vector<uint32_t>>> queryIndexFromFile(
      const std::string& query_file) {
    auto query_data = loadDataInMemory(query_file, 0);

    std::vector<std::vector<std::vector<uint32_t>>> results;

    for (uint32_t batch_index = 0; batch_index < query_data->len();
         batch_index++) {
      auto query_result = _flash_index->queryBatch(
          /* batch = */ query_data->at(batch_index),
          /* top_k = */ Indexer::TOP_K,
          /* pad_zeros = */ true);

      results.push_back(query_result);
    }

    return results;
  }

  std::vector<std::vector<uint32_t>> querySingle(const std::string& query) {
    auto featurized_query_vector = featurizeSingleQuery(query);

    std::vector<std::vector<uint32_t>> query_result = _flash_index->queryBatch(
        /* batch = */ BoltBatch(std::move(featurized_query_vector)),
        /* top_k = */ Indexer::TOP_K, /* pad_zeros = */ false);

    return query_result;
  }

 private:
  dataset::BoltDatasetPtr loadDataInMemory(
      const std::string& file_name, uint32_t correct_query_column_index) const {
    dataset::TextBlockPtr char_trigram_block =
        dataset::CharKGramTextBlock::make(
            /* col = */ correct_query_column_index, /* k = */ 3,
            /* dim = */ _dimension_for_encodings);

    dataset::TextBlockPtr char_four_gram = dataset::CharKGramTextBlock::make(
        /* col = */ correct_query_column_index, /* k = */ 4,
        /* dim = */ _dimension_for_encodings);

    std::vector<std::shared_ptr<dataset::Block>> input_blocks{
        char_trigram_block, char_four_gram};

    auto data_loader = dataset::StreamingGenericDatasetLoader(
        /* filename = */ file_name, /* input_blocks = */ input_blocks,
        /* label_blocks = */ {},
        /* batch_size = */ _flash_index_config->batch_size());

    auto [data, _] = data_loader.loadInMemory();

    return data;
  }

  std::vector<BoltVector> featurizeSingleQuery(const std::string& query) const {
    dataset::TextBlockPtr char_trigram_block =
        dataset::CharKGramTextBlock::make(
            /* col = */ 0, /* k = */ 3, /* dim = */ _dimension_for_encodings);
    dataset::TextBlockPtr char_four_gram = dataset::CharKGramTextBlock::make(
        /* col = */ 0, /* k = */ 4, /* dim = */ _dimension_for_encodings);

    std::vector<std::shared_ptr<dataset::Block>> input_blocks{
        char_trigram_block, char_four_gram};

    auto batch_processor = dataset::GenericBatchProcessor(input_blocks, {});

    BoltVector output_vector;
    std::vector<std::string_view> input_vector{
        std::string_view(query.data(), query.length())};
    if (auto exception =
            batch_processor.makeInputVector(input_vector, output_vector)) {
      std::rethrow_exception(exception);
    }
    return {std::move(output_vector)};
  }

  std::shared_ptr<IndexerConfig> _flash_index_config;

  std::unique_ptr<Flash<uint32_t>> _flash_index;
  uint32_t _dimension_for_encodings;

  // private constructor for cereal
  Indexer() {}

  friend class cereal::access;
  template <class Archive>
  void serialize(Archive& archive) {
    archive(_flash_index_config, _flash_index, _dimension_for_encodings);
  }
};

using IndexerPtr = std::shared_ptr<Indexer>;

}  // namespace thirdai::bolt