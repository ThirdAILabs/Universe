#pragma once

#include <cereal/access.hpp>
#include <cereal/archives/binary.hpp>
#include <cereal/types/memory.hpp>
#include <cereal/types/optional.hpp>
#include <cereal/types/unordered_map.hpp>
#include <cereal/types/vector.hpp>
#include <hashing/src/DWTA.h>
#include <hashing/src/DensifiedMinHash.h>
#include <hashing/src/FastSRP.h>
#include <hashing/src/MinHash.h>
#include <dataset/src/DataLoader.h>
#include <dataset/src/Datasets.h>
#include <dataset/src/StreamingGenericDatasetLoader.h>
#include <dataset/src/batch_processors/GenericBatchProcessor.h>
#include <dataset/src/blocks/BlockInterface.h>
#include <dataset/src/blocks/Text.h>
#include <dataset/src/utils/TextEncodingUtils.h>
#include <exceptions/src/Exceptions.h>
#include <search/src/Flash.h>
#include <fstream>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace thirdai::bolt {

using thirdai::search::Flash;

class QueryCandidateGeneratorConfig {
 public:
  QueryCandidateGeneratorConfig(std::string hash_function, uint32_t num_tables,
                                uint32_t hashes_per_table, uint32_t range,
                                std::vector<uint32_t> n_grams,
                                uint32_t source_column_index = 0,
                                uint32_t target_column_index = 0,
                                bool has_incorrect_queries = false,
                                uint32_t batch_size = 10000)
      : _hash_function(std::move(hash_function)),
        _num_tables(num_tables),
        _hashes_per_table(hashes_per_table),
        _batch_size(batch_size),
        _range(range),
        _n_grams(std::move(n_grams)),
        _has_incorrect_queries(has_incorrect_queries),
        _source_column_index(source_column_index),
        _target_column_index(target_column_index) {}

  // Overloaded operator mainly for testing
  bool operator==(const QueryCandidateGeneratorConfig& rhs) const {
    return this->_hash_function == rhs._hash_function &&
           this->_num_tables == rhs._num_tables &&
           this->_hashes_per_table == rhs._hashes_per_table &&
           this->_source_column_index == rhs._source_column_index && 
           this->_target_column_index == rhs._target_column_index && 
           this->_batch_size == rhs._batch_size && this->_range == rhs._range &&
           this->_n_grams == rhs._n_grams &&
           this->_has_incorrect_queries == rhs._has_incorrect_queries;
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
    auto hash_function = thirdai::utils::lower(_hash_function);

    if (hash_function == "minhash") {
      return std::make_shared<hashing::MinHash>(_hashes_per_table, _num_tables,
                                                _range);
    }
    if (hash_function == "densifiedminhash") {
      return std::make_shared<hashing::DensifiedMinHash>(_hashes_per_table,
                                                         _num_tables, _range);
    }
    throw exceptions::NotImplemented(
        "Unsupported Hash Function. Supported Hash Functions: "
        "DensifiedMinHash, MinHash.");
  }

  constexpr uint32_t batchSize() const { return _batch_size; }
  constexpr bool hasIncorrectQueries() const { return _has_incorrect_queries; }
  constexpr uint32_t sourceColumnIndex() const { return _source_column_index; }
  constexpr uint32_t targetColumnIndex() const { return _target_column_index; }

  std::vector<uint32_t> nGrams() const { return _n_grams; }

 private:
  std::string _hash_function;
  uint32_t _num_tables;
  uint32_t _hashes_per_table;

  uint32_t _batch_size;
  uint32_t _range;
  std::vector<uint32_t> _n_grams;

  // Identifies if the dataset contains pairs of correct and incorrect queries
  bool _has_incorrect_queries;
  uint32_t _source_column_index;
  uint32_t _target_column_index;

  // Private constructor for cereal
  QueryCandidateGeneratorConfig() {}

  friend class cereal::access;
  template <class Archive>
  void serialize(Archive& archive) {
    archive(_hash_function, _num_tables, _hashes_per_table, _batch_size, _range,
            _n_grams, _has_incorrect_queries);
  }
};

using QueryCandidateGeneratorConfigPtr =
    std::shared_ptr<QueryCandidateGeneratorConfig>;

class QueryCandidateGenerator {
 public:
  static QueryCandidateGenerator make(
      QueryCandidateGeneratorConfigPtr query_candidate_generator_config) {
    return QueryCandidateGenerator(std::move(query_candidate_generator_config));
  }

  static QueryCandidateGenerator buildGeneratorFromSerializedConfig(
      const std::string& config_file_name) {
    auto query_candidate_generator_config =
        QueryCandidateGeneratorConfig::load(config_file_name);

    return QueryCandidateGenerator::make(query_candidate_generator_config);
  }

  void save(const std::string& file_name) {
    std::ofstream filestream =
        dataset::SafeFileIO::ofstream(file_name, std::ios::binary);

    cereal::BinaryOutputArchive output_archive(filestream);
    output_archive(*this);
  }

  static std::shared_ptr<QueryCandidateGenerator> load(
      const std::string& file_name) {
    std::ifstream filestream =
        dataset::SafeFileIO::ifstream(file_name, std::ios::binary);

    cereal::BinaryInputArchive input_archive(filestream);
    std::shared_ptr<QueryCandidateGenerator> deserialized_generator(
        new QueryCandidateGenerator());
    input_archive(*deserialized_generator);

    return deserialized_generator;
  }

  /**
   * @brief Builds a Flash index by reading from a CSV file
   * containing queries.
   * If the `has_incorrect_queries` flag is set in the
   * QueryCandidateGeneratorConfig, the input CSV file is expected to contain
   * both correct (first column) and incorrect queries (second column).
   * Otherwise, the file is expected to have only correct queries in one
   * column.
   *
   * @param file_name
   */
  void buildFlashIndex(const std::string& file_name) {
    auto labels = getQueryLabels(
        file_name, _query_generator_config->hasIncorrectQueries());

    auto data_loader = getDatasetLoader(file_name, /* evaluate = */ false);
    auto [data, _] = data_loader->loadInMemory();

    if (!_flash_index) {
      _flash_index = std::make_unique<Flash<uint32_t>>(
          _query_generator_config->getHashFunction());
    }

    _flash_index->addDataset(*data, labels);
  }

  /**
   * @brief Given a vector of queries, returns top k generated queries
   * by the flash index for each of the queries. For instance, if
   * queries is a vector of size n, then the output will be a vector
   * of size n, each of which is also a vector of size at most k.
   *
   * @param queries
   * @param top_k
   * @return A vector of suggested queries

   */
  std::vector<std::vector<std::string>> queryFromList(
      const std::vector<std::string>& queries, uint32_t top_k) {
    if (!_flash_index) {
      throw exceptions::QueryCandidateGeneratorException(
          "Attempting to Generate Candidate Queries without Training the "
          "Generator.");
    }
    std::vector<BoltVector> featurized_queries(queries.size());
    for (uint32_t query_index = 0; query_index < queries.size();
         query_index++) {
      featurized_queries[query_index] =
          featurizeSingleQuery(queries[query_index]);
    }
    std::vector<std::vector<uint32_t>> candidate_query_labels =
        _flash_index->queryBatch(
            /* batch = */ BoltBatch(std::move(featurized_queries)),
            /* top_k = */ top_k,
            /* pad_zeros = */ false);

    std::vector<std::vector<std::string>> outputs;
    outputs.reserve(queries.size());

    for (auto& candidate_query_label_vector : candidate_query_labels) {
      auto top_k_candidates =
          getQueryCandidatesAsStrings(candidate_query_label_vector);

      outputs.emplace_back(std::move(top_k_candidates));
    }
    return outputs;
  }

  std::unordered_map<std::string, uint32_t> getQueriesToLabelsMap() const {
    return _queries_to_labels_map;
  }

 private:
  explicit QueryCandidateGenerator(
      QueryCandidateGeneratorConfigPtr query_candidate_generator_config)
      : _query_generator_config(std::move(query_candidate_generator_config)),
        _dimension_for_encodings(
            dataset::TextEncodingUtils::DEFAULT_TEXT_ENCODING_DIM) {
    std::vector<dataset::BlockPtr> training_input_blocks;

    if (_query_generator_config->hasIncorrectQueries()) {
      training_input_blocks = constructInputBlocks(
          _query_generator_config->nGrams(),
          /* column_index = */ _query_generator_config->sourceColumnIndex());
      auto inference_input_blocks = constructInputBlocks(
          _query_generator_config->nGrams(),
          /* column_index = */ _query_generator_config->targetColumnIndex());

      _training_batch_processor =
          std::make_shared<dataset::GenericBatchProcessor>(
              training_input_blocks, std::vector<dataset::BlockPtr>{});

      _inference_batch_processor =
          std::make_shared<dataset::GenericBatchProcessor>(
              inference_input_blocks, std::vector<dataset::BlockPtr>{});

    } else {
      assert(_query_generator_config->sourceColumnIndex() ==
             _query_generator_config->_target_column_index());
      training_input_blocks = constructInputBlocks(
          _query_generator_config->nGrams(),
          /* column_index = */ _query_generator_config->targetColumnIndex());

      _training_batch_processor =
          std::make_shared<dataset::GenericBatchProcessor>(
              training_input_blocks, std::vector<dataset::BlockPtr>{});

      _inference_batch_processor = _training_batch_processor;
    }
  }

  std::vector<dataset::BlockPtr> constructInputBlocks(
      const std::vector<uint32_t>& n_grams, uint32_t column_index) const {
    std::vector<dataset::BlockPtr> input_blocks;
    input_blocks.reserve(n_grams.size());

    for (auto n_gram : n_grams) {
      input_blocks.emplace_back(dataset::CharKGramTextBlock::make(
          /* col = */ column_index,
          /* k = */ n_gram,
          /* dim = */ _dimension_for_encodings));
    }

    return input_blocks;
  }

  std::vector<std::string> getQueryCandidatesAsStrings(
      const std::vector<uint32_t>& query_labels) {
    std::vector<std::string> output_strings;
    output_strings.reserve(query_labels.size());

    for (const auto& query_label : query_labels) {
      output_strings.push_back(std::move(_labels_to_queries_map[query_label]));
    }
    return output_strings;
  }

  /**
   * @brief Constructs a mapping from correct queries to the labels. This allows
   * us to convert the output of queryBatch() into strings representing
   * correct queries.
   *
   * @param file_name: File containing queries (Expected to be a CSV file).
   * @param has_incorrect_queries: Identifies if the input CSV file contains
   *        pairs of correct and incorrect queries.
   *
   * @return Labels for each batch
   */
  std::vector<std::vector<uint32_t>> getQueryLabels(
      const std::string& file_name, bool has_incorrect_queries) {
    std::vector<std::vector<uint32_t>> labels;

    try {
      std::ifstream input_file_stream =
          dataset::SafeFileIO::ifstream(file_name, std::ios::in);

      std::vector<uint32_t> current_batch_labels;
      std::string row;

      while (std::getline(input_file_stream, row)) {
        std::string correct_query;

        if (has_incorrect_queries) {
          correct_query =
              std::string(dataset::ProcessorUtils::parseCsvRow(row, ',')[0]);
        } else {
          correct_query = row;
        }

        if (!_queries_to_labels_map.count(correct_query)) {
          _queries_to_labels_map[correct_query] = _labels_to_queries_map.size();
          _labels_to_queries_map.push_back(correct_query);
        }

        // Add the corresponding label to the current batch
        current_batch_labels.push_back(
            _queries_to_labels_map.at(correct_query));

        if (current_batch_labels.size() ==
            _query_generator_config->batchSize()) {
          labels.push_back(std::move(current_batch_labels));
        }
      }
      input_file_stream.close();

      // Add remainig labels if present. This will happen
      // if the entire dataset fits in one batch or the last
      // batch has fewer elements than the batch size.
      if (!current_batch_labels.empty()) {
        labels.push_back(std::move(current_batch_labels));
      }

    } catch (const std::ifstream::failure& exception) {
      throw std::invalid_argument("Invalid input file name.");
    }

    return labels;
  }

  BoltVector featurizeSingleQuery(const std::string& query) const {
    BoltVector output_vector;
    std::vector<std::string_view> input_vector{
        std::string_view(query.data(), query.length())};
    if (auto exception = _inference_batch_processor->makeInputVector(
            input_vector, output_vector)) {
      std::rethrow_exception(exception);
    }
    return output_vector;
  }

  std::unique_ptr<dataset::StreamingGenericDatasetLoader> getDatasetLoader(
      const std::string& file_name) {
    auto file_data_loader = dataset::SimpleFileDataLoader::make(
        file_name, _query_generator_config->batchSize());

    return std::make_unique<dataset::StreamingGenericDatasetLoader>(
                          file_data_loader, _training_batch_processor);
  }

  std::shared_ptr<QueryCandidateGeneratorConfig> _query_generator_config;
  uint32_t _dimension_for_encodings;

  std::unique_ptr<Flash<uint32_t>> _flash_index;
  std::shared_ptr<dataset::GenericBatchProcessor> _training_batch_processor;
  std::shared_ptr<dataset::GenericBatchProcessor> _inference_batch_processor;

  /**
   * Maintains a mapping from the assigned labels to the original
   * queries loaded from a CSV file. Each unique query in the input
   * CSV file is assigned a unique label in an ascending order. This is
   * a vector instead of an unordered map because queries are
   * assigned labels sequentially.
   */
  std::vector<std::string> _labels_to_queries_map;

  std::unordered_map<std::string, uint32_t> _queries_to_labels_map;

  // private constructor for cereal
  QueryCandidateGenerator() {}

  friend class cereal::access;
  template <class Archive>
  void serialize(Archive& archive) {
    archive(_query_generator_config, _dimension_for_encodings, _flash_index,
            _training_batch_processor, _inference_batch_processor,
            _labels_to_queries_map, _queries_to_labels_map);
  }
};

using QueryCandidateGeneratorPtr = std::shared_ptr<QueryCandidateGenerator>;

}  // namespace thirdai::bolt