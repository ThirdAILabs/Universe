#pragma once

#include <cereal/access.hpp>
#include <cereal/archives/binary.hpp>
#include <cereal/types/memory.hpp>
#include <cereal/types/optional.hpp>
#include <cereal/types/unordered_map.hpp>
#include <cereal/types/vector.hpp>
#include <hashing/src/DensifiedMinHash.h>
#include <hashing/src/MinHash.h>
#include <auto_ml/src/deployment_config/dataset_configs/udt/ColumnNumberMap.h>
#include <auto_ml/src/deployment_config/dataset_configs/udt/UDTDatasetFactory.h>
#include <dataset/src/DataLoader.h>
#include <dataset/src/Datasets.h>
#include <dataset/src/StreamingGenericDatasetLoader.h>
#include <dataset/src/batch_processors/GenericBatchProcessor.h>
#include <dataset/src/batch_processors/ProcessorUtils.h>
#include <dataset/src/blocks/BlockInterface.h>
#include <dataset/src/blocks/Text.h>
#include <dataset/src/utils/TextEncodingUtils.h>
#include <exceptions/src/Exceptions.h>
#include <search/src/Flash.h>
#include <fstream>
#include <limits>
#include <memory>
#include <optional>
#include <ostream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace thirdai::bolt {

using thirdai::automl::deployment::ColumnNumberMap;
using thirdai::automl::deployment::ColumnNumberMapPtr;
using thirdai::search::Flash;

class QueryCandidateGeneratorConfig {
  static inline const uint32_t DEFAULT_BATCH_SIZE = 1000;
  static inline const char* DEFAULT_HASH_FUNCTION = "minhash";
  static inline const uint32_t DEFAULT_NUM_TABLES = 128;
  static inline const uint32_t DEFAULT_HASHES_PER_TABLE = 3;
  static inline const uint32_t DEFAULT_RESERVOIR_SIZE = 512;
  static inline const uint32_t DEFAULT_RANGE = 100000;

 public:
  QueryCandidateGeneratorConfig(
      const std::string& hash_function, uint32_t num_tables,
      uint32_t hashes_per_table, uint32_t range, std::vector<uint32_t> n_grams,
      std::optional<uint32_t> reservoir_size, uint32_t source_column_index,
      uint32_t target_column_index, std::string file_name,
      uint32_t batch_size = 10000,
      uint32_t default_text_encoding_dim = std::numeric_limits<uint32_t>::max())
      : _num_tables(num_tables),
        _hashes_per_table(hashes_per_table),
        _batch_size(batch_size),
        _range(range),
        _n_grams(std::move(n_grams)),
        _reservoir_size(reservoir_size),
        _source_column_index(source_column_index),
        _target_column_index(target_column_index),
        _default_text_encoding_dim(default_text_encoding_dim),
        _queries_file_name(std::move(file_name)) {
    _hash_function = getHashFunction(
        /* hash_function = */ thirdai::utils::lower(hash_function));
  }

  // Overloaded operator mainly for testing
  bool operator==(const QueryCandidateGeneratorConfig& rhs) const {
    return this->_hash_function->getName() == rhs._hash_function->getName() &&
           this->_num_tables == rhs._num_tables &&
           this->_hashes_per_table == rhs._hashes_per_table &&
           this->_source_column_index == rhs._source_column_index &&
           this->_target_column_index == rhs._target_column_index &&
           this->_batch_size == rhs._batch_size && this->_range == rhs._range &&
           this->_n_grams == rhs._n_grams &&
           this->_reservoir_size == rhs._reservoir_size;
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

  constexpr uint32_t batchSize() const { return _batch_size; }

  std::optional<uint32_t> reservoirSize() const { return _reservoir_size; }

  constexpr uint32_t sourceColumnIndex() const { return _source_column_index; }

  constexpr uint32_t targetColumnIndex() const { return _target_column_index; }

  constexpr uint32_t defaultTextEncodingDim() const {
    return _default_text_encoding_dim;
  }

  std::string queriesFileName() const { return _queries_file_name; }

  static std::shared_ptr<QueryCandidateGeneratorConfig> fromDefault(
      const std::string& file_name, const std::string& source_column,
      const std::string& target_column) {
    auto column_number_map =
        getColumnNumberMap(/* file_name = */ file_name, /* delimiter = */ ',');
    uint32_t source_column_index =
        column_number_map->at(/* col_name = */ source_column);
    uint32_t target_column_index =
        column_number_map->at(/* col_name = */ target_column);

    auto generator_config = QueryCandidateGeneratorConfig(
        /* hash_function = */ "minhash",
        /* num_tables = */ DEFAULT_NUM_TABLES,
        /* hashes_per_table = */ DEFAULT_HASHES_PER_TABLE,
        /* range = */ DEFAULT_RANGE,
        /* n_grams = */ {3, 4},
        /* reservoir_size */ DEFAULT_RESERVOIR_SIZE,
        /* source_column_index = */ source_column_index,
        /* target_column_index = */ target_column_index,
        /* file_name = */ file_name);

    return std::make_shared<QueryCandidateGeneratorConfig>(generator_config);
  }

  // static std::shared_ptr<QueryCandidateGeneratorConfig> fromDefault(
  //     const uint32_t& source_column_index,
  //     const uint32_t& target_column_index) {
  //   auto generator_config = QueryCandidateGeneratorConfig(
  //       /* hash_function = */ "minhash",
  //       /* num_tables = */ DEFAULT_NUM_TABLES,
  //       /* hashes_per_table = */ DEFAULT_HASHES_PER_TABLE,
  //       /* range = */ DEFAULT_RANGE,
  //       /* n_grams = */ {3, 4},
  //       /* reservoir_size */ DEFAULT_RESERVOIR_SIZE,
  //       /* source_column_index = */ source_column_index,
  //       /* target_column_index = */ target_column_index);

  //   return std::make_shared<QueryCandidateGeneratorConfig>(generator_config);
  // }

  std::shared_ptr<hashing::HashFunction> hashFunction() const {
    return _hash_function;
  }

  std::vector<uint32_t> nGrams() const { return _n_grams; }

 private:
  static ColumnNumberMapPtr getColumnNumberMap(const std::string& file_name,
                                               char delimiter) {
    auto file_data_loader = dataset::SimpleFileDataLoader::make(
        /* filename = */ file_name,
        /* target_batch_size = */ DEFAULT_BATCH_SIZE);
    auto file_header = file_data_loader->nextLine();
    if (!file_header) {
      throw std::invalid_argument(
          "The input dataset file must have a valid header with column names.");
    }
    return std::make_shared<ColumnNumberMap>(*file_header, delimiter);
  }

  std::shared_ptr<hashing::HashFunction> getHashFunction(
      const std::string& hash_function) {
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

  std::shared_ptr<hashing::HashFunction> _hash_function;
  uint32_t _num_tables;
  uint32_t _hashes_per_table;

  uint32_t _batch_size;
  uint32_t _range;
  std::vector<uint32_t> _n_grams;

  // Identifies if the dataset contains pairs of correct and incorrect queries
  std::optional<uint32_t> _reservoir_size;

  uint32_t _source_column_index;
  uint32_t _target_column_index;
  uint32_t _default_text_encoding_dim;
  std::string _queries_file_name;

  // Private constructor for cereal
  QueryCandidateGeneratorConfig() {}

  friend class cereal::access;
  template <class Archive>
  void serialize(Archive& archive) {
    archive(_hash_function, _num_tables, _hashes_per_table, _batch_size, _range,
            _n_grams, _reservoir_size, _source_column_index,
            _target_column_index, _default_text_encoding_dim,
            _queries_file_name);
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

  static QueryCandidateGenerator buildGeneratorFromDefaultConfig(
      const std::string& file_name, const std::string& source_column,
      const std::string& target_column, const std::string& dataset_size) {
    (void)dataset_size;
    auto config = QueryCandidateGeneratorConfig::fromDefault(
        /* file_name = */ file_name, /* source_column = */ source_column,
        /* target_column = */ target_column);

    auto generator = QueryCandidateGenerator::make(config);
    return generator;
  }

  // static QueryCandidateGenerator buildGeneratorFromDefaultConfig(
  //     const uint32_t& source_column_index, const uint32_t&
  //     target_column_index, const std::string& dataset_size) {
  //   (void)dataset_size;
  //   auto config = QueryCandidateGeneratorConfig::fromDefault(
  //       /* source_column_index = */ source_column_index,
  //       /* target_column_index = */ target_column_index);
  //   auto generator = QueryCandidateGenerator::make(config);

  //   return generator;
  // }

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
  void buildFlashIndex() {
    auto file_name = _query_generator_config->queriesFileName();
    auto labels = getQueryLabels(file_name);

    auto data_loader = getDatasetLoader(file_name);
    auto [data, _] = data_loader->loadInMemory();

    if (!_flash_index) {
      auto hash_function = _query_generator_config->hashFunction();
      if (_query_generator_config->reservoirSize().has_value()) {
        _flash_index = std::make_unique<Flash<uint32_t>>(
            hash_function, _query_generator_config->reservoirSize().value());
      } else {
        _flash_index = std::make_unique<Flash<uint32_t>>(hash_function);
      }
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

  /**
   * @brief Returns a list of recommended queries and Computes Recall at K
   * (R1@K).
   *
   * @param file_name: CSV file expected to have correct queries in column 0,
   * and incorrect queries in column 1.
   * @return Recommended queries
   */
  std::vector<std::vector<std::string>> evaluateOnFile(
      const std::string& file_name, uint32_t top_k) {
    if (!_flash_index) {
      throw exceptions::QueryCandidateGeneratorException(
          "Attempting to Evaluate the Generator without Training.");
    }
    auto data_loader = getDatasetLoader(file_name);
    auto [data, _] = data_loader->loadInMemory();

    std::vector<std::vector<std::string>> output_queries;
    for (const auto& batch : *data) {
      std::vector<std::vector<uint32_t>> candidate_query_labels =
          _flash_index->queryBatch(
              /* batch = */ batch,
              /* top_k = */ top_k,
              /* pad_zeros = */ false);

      for (auto& candidate_query_label_vector : candidate_query_labels) {
        auto top_k = getQueryCandidatesAsStrings(candidate_query_label_vector);
        output_queries.push_back(std::move(top_k));
      }
    }

    if (_query_generator_config->sourceColumnIndex() !=
        _query_generator_config->targetColumnIndex()) {
      std::vector<std::string> correct_queries =
          dataset::ProcessorUtils::aggregateSingleColumnCsvRows(
              file_name, /* column_index = */ _query_generator_config
                             ->targetColumnIndex());

      computeRecallAtK(/* correct_queries = */ correct_queries,
                       /* generated_queries = */ output_queries,
                       /* K = */ top_k);
    }
    return output_queries;
  }

  std::unordered_map<std::string, uint32_t> getQueriesToLabelsMap() const {
    return _queries_to_labels_map;
  }

 private:
  explicit QueryCandidateGenerator(
      QueryCandidateGeneratorConfigPtr query_candidate_generator_config)
      : _query_generator_config(std::move(query_candidate_generator_config)) {
    std::vector<dataset::BlockPtr> training_input_blocks;

    auto inference_input_blocks =
        constructInputBlocks(_query_generator_config->nGrams(),
                             /* column_index = */ 0);

    training_input_blocks = constructInputBlocks(
        _query_generator_config->nGrams(),
        /* column_index = */ _query_generator_config->sourceColumnIndex());

    _training_batch_processor =
        std::make_shared<dataset::GenericBatchProcessor>(
            /* input_blocks = */ training_input_blocks,
            /* label_blocks = */ std::vector<dataset::BlockPtr>{},
            /* has_header = */ false, /* delimiter = */ ',');

    _inference_batch_processor =
        std::make_shared<dataset::GenericBatchProcessor>(
            /* input_blocks = */ inference_input_blocks,
            /* labels_blocks = */ std::vector<dataset::BlockPtr>{},
            /* has_header = */ false, /* delimiter = */ ',');
  }

  std::vector<dataset::BlockPtr> constructInputBlocks(
      const std::vector<uint32_t>& n_grams, uint32_t column_index) const {
    std::vector<dataset::BlockPtr> input_blocks;
    input_blocks.reserve(n_grams.size());

    for (auto n_gram : n_grams) {
      input_blocks.emplace_back(dataset::CharKGramTextBlock::make(
          /* col = */ column_index,
          /* k = */ n_gram,
          /* dim = */ _query_generator_config->defaultTextEncodingDim()));
    }

    return input_blocks;
  }

  std::vector<std::string> getQueryCandidatesAsStrings(
      const std::vector<uint32_t>& query_labels) {
    std::vector<std::string> output_strings;
    output_strings.reserve(query_labels.size());

    for (const auto& query_label : query_labels) {
      output_strings.push_back(_labels_to_queries_map[query_label]);
    }
    return output_strings;
  }

  /**
   * @brief computes recall at K (R1@K)
   *
   * @param correct_queries: Vector of correct queries of size n.
   * @param generated_queries: Vector of generated queries by the flash index.
   * This is expected to be of size n and each element in the vector is also a
   * vector of size k.
   * @return recall
   */
  static void computeRecallAtK(
      const std::vector<std::string>& correct_queries,
      const std::vector<std::vector<std::string>>& generated_queries, int K) {
    assert(correct_queries.size() == generated_queries.size());
    uint64_t correct_results = 0;

    for (uint64_t query_index = 0; query_index < correct_queries.size();
         query_index++) {
      const auto& top_k_queries = generated_queries[query_index];
      const auto& correct_query = correct_queries[query_index];
      if (std::find(top_k_queries.begin(), top_k_queries.end(),
                    correct_query) != top_k_queries.end()) {
        correct_results += 1;
      }
    }
    auto recall = (correct_results * 1.0) / correct_queries.size();

    std::cout << "Recall@" << K << " = " << recall << std::endl;
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
      const std::string& file_name) {
    std::vector<std::vector<uint32_t>> labels;

    try {
      std::ifstream input_file_stream =
          dataset::SafeFileIO::ifstream(file_name, std::ios::in);

      std::vector<uint32_t> current_batch_labels;
      std::string row;

      while (std::getline(input_file_stream, row)) {
        std::string correct_query =
            std::string(dataset::ProcessorUtils::parseCsvRow(
                row, ',')[_query_generator_config->targetColumnIndex()]);

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

  // Private constructor for cereal
  QueryCandidateGenerator() {}

  friend class cereal::access;
  template <class Archive>
  void serialize(Archive& archive) {
    archive(_query_generator_config, _flash_index, _training_batch_processor,
            _inference_batch_processor, _labels_to_queries_map,
            _queries_to_labels_map);
  }
};

using QueryCandidateGeneratorPtr = std::shared_ptr<QueryCandidateGenerator>;

}  // namespace thirdai::bolt