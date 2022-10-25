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
#include <generator/src/Flash.h>
#include <fstream>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace thirdai::bolt {

class QueryCandidateGeneratorConfig {
 public:
  QueryCandidateGeneratorConfig(std::string hash_function, uint32_t num_tables,
                                uint32_t hashes_per_table, uint32_t input_dim,
                                uint32_t top_k, std::vector<uint32_t> n_grams,
                                bool has_incorrect_queries = false,
                                uint32_t batch_size = 10000,
                                uint32_t range = 1000000)
      : _hash_function(std::move(hash_function)),
        _num_tables(num_tables),
        _hashes_per_table(hashes_per_table),
        _input_dim(input_dim),
        _top_k(top_k),
        _batch_size(batch_size),
        _range(range),
        _n_grams(std::move(n_grams)),
        _has_incorrect_queries(has_incorrect_queries) {}

  // overloaded operator mainly for testing
  bool operator==(QueryCandidateGeneratorConfig* rhs) const {
    return this->_hash_function == rhs->_hash_function &&
           this->_num_tables == rhs->_num_tables &&
           this->_hashes_per_table == rhs->_hashes_per_table &&
           this->_input_dim == rhs->_input_dim && this->_top_k == rhs->_top_k &&
           this->_batch_size == rhs->_batch_size &&
           this->_range == rhs->_range && this->_n_grams == rhs->_n_grams &&
           this->_has_incorrect_queries == rhs->_has_incorrect_queries;
  }

  void save(const std::string& config_file_name) const {
    std::ofstream filestream =
        dataset::SafeFileIO::ofstream(config_file_name, std::ios::binary);

    cereal::BinaryOutputArchive output_archive(filestream);
    output_archive(*this);
  }

  static std::shared_ptr<QueryCandidateGeneratorConfig> load(
      const std::string& config_file_name) {
    std::ifstream filestream =
        dataset::SafeFileIO::ifstream(config_file_name, std::ios::binary);

    cereal::BinaryInputArchive input_archive(filestream);
    std::shared_ptr<QueryCandidateGeneratorConfig> deserialized_config(
        new QueryCandidateGeneratorConfig());
    input_archive(*deserialized_config);

    return deserialized_config;
  }

  std::shared_ptr<hashing::HashFunction> getHashFunction() const {
    if (_hash_function == "DensifiedMinHash") {
      return std::make_shared<hashing::DensifiedMinHash>(_hashes_per_table,
                                                         _num_tables, _range);
    }
    if (_hash_function == "DWTA") {
      return std::make_shared<hashing::DWTAHashFunction>(
          _input_dim, _hashes_per_table, _num_tables, _range);
    }
    if (_hash_function == "FastSRP") {
      return std::make_shared<hashing::FastSRP>(_input_dim, _hashes_per_table,
                                                _num_tables);
    }
    throw exceptions::NotImplemented("Unsupported Hash Function");
  }

  constexpr uint32_t batchSize() const { return _batch_size; }
  constexpr uint32_t topK() const { return _top_k; }
  constexpr bool hasIncorrectQueries() const { return _has_incorrect_queries; }
  std::vector<uint32_t> nGrams() const { return _n_grams; }

 private:
  std::string _hash_function;
  uint32_t _num_tables;
  uint32_t _hashes_per_table;

  uint32_t _input_dim;
  uint32_t _top_k;
  uint32_t _batch_size;
  uint32_t _range;
  std::vector<uint32_t> _n_grams;

  // Identifies if the dataset contains pairs of correct and incorrect queries
  bool _has_incorrect_queries;

  // Private constructor for cereal
  QueryCandidateGeneratorConfig() {}

  friend class cereal::access;
  template <class Archive>
  void serialize(Archive& archive) {
    archive(_hash_function, _num_tables, _hashes_per_table, _input_dim, _top_k,
            _batch_size, _range, _n_grams, _has_incorrect_queries);
  }
};

using QueryCandidateGeneratorConfigPtr =
    std::shared_ptr<QueryCandidateGeneratorConfig>;

class QueryCandidateGenerator {
 public:
  static QueryCandidateGenerator make(
      QueryCandidateGeneratorConfigPtr flash_QueryCandidateGenerator_config) {
    return QueryCandidateGenerator(
        std::move(flash_QueryCandidateGenerator_config));
  }

  static QueryCandidateGenerator buildGeneratorFromSerializedConfig(
      const std::string& config_file_name) {
    auto flash_QueryCandidateGenerator_config =
        QueryCandidateGeneratorConfig::load(config_file_name);

    return QueryCandidateGenerator::make(flash_QueryCandidateGenerator_config);
  }

  /**
   * @brief Builds a Flash index by reading from a CSV file
   * containing queries.
   * If the `has_incorrect_queries` flag is set in the
   * QueryCandidateGeneratorConfig, the input CSV file is expected to contain
   * both correct (first column) and incorrect queries (second column).
   * Otherwise, the file is expected to have only correct queries in one column.
   *
   * @param file_name
   */
  void buildFlashIndex(const std::string& file_name) {
    buildIDToQueryMapping(file_name,
                          _query_generator_config->hasIncorrectQueries());

    auto data_loader = getUnlabeledDatasetLoader(file_name);
    auto [data, _] = data_loader->loadInMemory();

    _flash_index = std::make_unique<Flash<uint32_t>>(
        _query_generator_config->getHashFunction());
    _flash_index->addDataset(*data);
  }

  /**
   * @brief Given a vector of queries, returns top k generated queries
   * by the flash instance for each of the query. For instance, if
   * queries is a vector of size n, then the output will be a vector
   * of size n, each of which is also a vector of size k.
   *
   * @param queries
   * @return A vector of suggested queries

   */
  std::vector<std::vector<std::string>> queryFromList(
      const std::vector<std::string>& queries) {
    std::vector<std::vector<std::string>> outputs(queries.size());

#pragma omp parallel for default(none) shared(queries, outputs)
    for (uint32_t query_index = 0; query_index < queries.size();
         query_index++) {
      auto featurized_query_vector = featurizeSingleQuery(queries[query_index]);

      std::vector<std::vector<uint32_t>> suggested_query_ids =
          _flash_index->queryBatch(
              /* batch = */ BoltBatch(std::move(featurized_query_vector)),
              /* top_k = */ _query_generator_config->topK(),
              /* pad_zeros = */ true);

      std::vector<std::string> topk_queries(_query_generator_config->topK());
      for (uint32_t query_id_index = 0;
           query_id_index < _query_generator_config->topK(); query_id_index++) {
        auto suggested_query =
            _ids_to_queries_map.at(suggested_query_ids[0][query_id_index]);
        topk_queries[query_id_index] = std::move(suggested_query);
      }
      outputs[query_index] = std::move(topk_queries);
    }

    return outputs;
  }

 private:
  explicit QueryCandidateGenerator(
      QueryCandidateGeneratorConfigPtr flash_generator_config)
      : _query_generator_config(std::move(flash_generator_config)),
        _dimension_for_encodings(
            dataset::TextEncodingUtils::DEFAULT_TEXT_ENCODING_DIM),
        _input_blocks(constructInputBlocks(_query_generator_config->nGrams())),
        _batch_processor(std::make_shared<dataset::GenericBatchProcessor>(
            _input_blocks, std::vector<dataset::BlockPtr>{})) {}

  std::vector<dataset::BlockPtr> constructInputBlocks(
      const std::vector<uint32_t>& n_grams) const {
    uint8_t correct_query_column_index = 0;

    std::vector<dataset::BlockPtr> input_blocks(n_grams.size());

    for (auto n_gram : n_grams) {
      input_blocks.emplace_back(dataset::CharKGramTextBlock::make(
          /* col = */ correct_query_column_index,
          /* k = */ n_gram,
          /* dim = */ _dimension_for_encodings));
    }

    return input_blocks;
  }

  std::unordered_map<uint32_t, std::string> getIDToQueryMapping() const {
    return _ids_to_queries_map;
  }
  /**
   * @brief Constructs a mapping from IDs to correct queries. This allows
   * us to convert the output of queryBatch() into strings representing
   * correct queries.
   *
   * @param file_name: File containing queries (Expected to be a CSV file).
   */
  void buildIDToQueryMapping(const std::string& file_name,
                             bool has_incorrect_queries) {
    try {
      std::ifstream input_file_stream =
          dataset::SafeFileIO::ifstream(file_name, std::ios::in);

      uint32_t ids_counter = 0;
      std::string row;

      while (std::getline(input_file_stream, row)) {
        if (has_incorrect_queries) {
          auto parsed_row = dataset::ProcessorUtils::parseCsvRow(row, ',');
          auto correct_query = std::string(parsed_row[0]);

          if (!_query_to_ids_map.count(correct_query)) {
            _ids_to_queries_map[ids_counter] = correct_query;
            _query_to_ids_map[correct_query] = ids_counter;
          }
        } else {
          _ids_to_queries_map[ids_counter] = row;
        }
        ids_counter += 1;
      }
      input_file_stream.close();

    } catch (const std::ifstream::failure& exception) {
      throw std::invalid_argument("Invalid input file name.");
    }
  }

  std::vector<BoltVector> featurizeSingleQuery(const std::string& query) const {
    BoltVector output_vector;
    std::vector<std::string_view> input_vector{
        std::string_view(query.data(), query.length())};
    if (auto exception =
            _batch_processor->makeInputVector(input_vector, output_vector)) {
      std::rethrow_exception(exception);
    }
    return {std::move(output_vector)};
  }

  std::unique_ptr<dataset::StreamingGenericDatasetLoader>
  getUnlabeledDatasetLoader(const std::string& file_name) {
    auto file_data_loader = dataset::SimpleFileDataLoader::make(
        file_name, _query_generator_config->batchSize());

    return std::make_unique<dataset::StreamingGenericDatasetLoader>(
        file_data_loader, _batch_processor);
  }

  std::shared_ptr<QueryCandidateGeneratorConfig> _query_generator_config;
  uint32_t _dimension_for_encodings;

  std::unique_ptr<Flash<uint32_t>> _flash_index;

  std::vector<dataset::BlockPtr> _input_blocks;
  std::shared_ptr<dataset::GenericBatchProcessor> _batch_processor;

  // Maintains a mapping from the assigned IDs to the original
  // queries loaded from a CSV file. Each unique query in the input
  // CSV file is assigned a unique ID in an ascending order.
  std::unordered_map<uint32_t, std::string> _ids_to_queries_map;

  std::unordered_map<std::string, uint32_t> _query_to_ids_map;

  // private constructor for cereal
  QueryCandidateGenerator() {}

  friend class cereal::access;
  template <class Archive>
  void serialize(Archive& archive) {
    archive(_query_generator_config, _dimension_for_encodings, _flash_index,
            _input_blocks, _batch_processor, _ids_to_queries_map);
  }
};  // namespace thirdai::bolt

using QueryCandidateGeneratorPtr = std::shared_ptr<QueryCandidateGenerator>;

}  // namespace thirdai::bolt