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

class GeneratorConfig {
 public:
  GeneratorConfig(std::string hash_function, uint32_t num_tables,
                  uint32_t hashes_per_table, uint32_t input_dim,
                  uint32_t topk = 5, uint32_t batch_size = 100,
                  uint32_t range = 1000000)
      : _hash_function(std::move(hash_function)),
        _num_tables(num_tables),
        _hashes_per_table(hashes_per_table),
        _input_dim(input_dim),
        _topk(topk),
        _batch_size(batch_size),
        _range(range) {}

  // overloaded operator mainly for testing
  auto operator==(GeneratorConfig* rhs) const {
    return this->_hash_function == rhs->_hash_function &&
           this->_num_tables == rhs->_num_tables &&
           this->_hashes_per_table == rhs->_hashes_per_table &&
           this->_input_dim == rhs->_input_dim && this->_topk == rhs->_topk &&
           this->_batch_size == rhs->_batch_size && this->_range == rhs->_range;
  }

  void save(const std::string& config_file_name) const {
    std::ofstream filestream =
        dataset::SafeFileIO::ofstream(config_file_name, std::ios::binary);

    cereal::BinaryOutputArchive output_archive(filestream);
    output_archive(*this);
  }

  static std::shared_ptr<GeneratorConfig> load(
      const std::string& config_file_name) {
    std::ifstream filestream =
        dataset::SafeFileIO::ifstream(config_file_name, std::ios::binary);

    cereal::BinaryInputArchive input_archive(filestream);
    std::shared_ptr<GeneratorConfig> deserialized_config(new GeneratorConfig());
    input_archive(*deserialized_config);

    return deserialized_config;
  }

  std::shared_ptr<hashing::HashFunction> getHashFunction() const {
    if (_hash_function == "DensifiedMinHash") {
      auto* hash_function =
          new hashing::DensifiedMinHash(_hashes_per_table, _num_tables, _range);
      return std::make_shared<hashing::DensifiedMinHash>(*hash_function);
    }
    if (_hash_function == "DWTA") {
      auto* hash_function = new hashing::DWTAHashFunction(
          _input_dim, _hashes_per_table, _num_tables, _range);
      return std::make_shared<hashing::DWTAHashFunction>(*hash_function);
    }
    if (_hash_function == "FastSRP") {
      auto* hash_function =
          new hashing::FastSRP(_input_dim, _hashes_per_table, _num_tables);
      return std::make_shared<hashing::FastSRP>(*hash_function);
    }
    throw exceptions::NotImplemented("Unsupported Hash Function");
  }

  constexpr uint32_t batch_size() const { return _batch_size; }
  constexpr uint32_t topk() const { return _topk; }

 private:
  std::string _hash_function;
  uint32_t _num_tables;
  uint32_t _hashes_per_table;

  uint32_t _input_dim;
  uint32_t _topk;
  uint32_t _batch_size;
  uint32_t _range;

  // Private constructor for cereal
  GeneratorConfig() {}

  friend class cereal::access;
  template <class Archive>
  void serialize(Archive& archive) {
    archive(_hash_function, _input_dim, _batch_size, _num_tables, _range, _topk,
            _hashes_per_table);
  }
};

using GeneratorConfigPtr = std::shared_ptr<GeneratorConfig>;

class Generator : public std::enable_shared_from_this<Generator> {
 public:
  explicit Generator(GeneratorConfigPtr flash_generator_config)
      : _flash_generator_config(std::move(flash_generator_config)),
        _dimension_for_encodings(
            dataset::TextEncodingUtils::DEFAULT_TEXT_ENCODING_DIM) {}

  static Generator make(GeneratorConfigPtr flash_generator_config) {
    return Generator(std::move(flash_generator_config));
  }

  static Generator buildGeneratorFromSerializedConfig(
      const std::string& config_file_name) {
    auto flash_generator_config = GeneratorConfig::load(config_file_name);

    return Generator::make(flash_generator_config);
  }

  /**
   * @brief Builds a Flash object by reading from a CSV file
   * containing queries.
   * If the `has_incorrect_queries` flag is set, the input
   * CSV file is expected to contain both incorrect (first column)
   * and correct queries (second column). Otherwise, the file is
   * expected to have only correct queries in one column.
   *
   * @param file_name
   * @param has_incorrect_queries
   * @return std::shared_ptr<Generator>
   */
  std::shared_ptr<Generator> buildFlashGenerator(const std::string& file_name,
                                                 bool has_incorrect_queries) {
    buildIDsQueryMapping(file_name, has_incorrect_queries);

    uint8_t correct_query_column_index = 0;
    if (has_incorrect_queries) {
      correct_query_column_index = 1;
    }

    auto data_loader = getUnlabeledDatasetLoader(
        file_name,
        /* correct_query_column_index = */ correct_query_column_index);
    auto [data, _] = data_loader->loadInMemory();

    _flash_generator = std::make_unique<Flash<uint32_t>>(
        _flash_generator_config->getHashFunction());
    _flash_generator->addDataset(*data);

    return shared_from_this();
  }

  std::vector<std::vector<std::vector<std::string>>> queryFromFile(
      const std::string& query_file) {
    auto data_loader = getUnlabeledDatasetLoader(
        query_file, /* correct_query_column_index = */ 0);

    auto query_data = std::get<0>(data_loader->loadInMemory());

    std::vector<std::vector<std::vector<std::string>>> results;

#pragma omp parallel for default(none) shared(query_data, results)
    for (uint32_t batch_index = 0; batch_index < query_data->len();
         batch_index++) {
      auto batch_suggested_ids = _flash_generator->queryBatch(
          /* batch = */ query_data->at(batch_index),
          /* top_k = */ _flash_generator_config->topk(),
          /* pad_zeros = */ true);

      std::vector<std::vector<std::string>> batch_suggestions;
      std::vector<std::string> topk_queries;
      for (auto& vector_suggested_ids : batch_suggested_ids) {
        for (auto suggested_query_id : vector_suggested_ids) {
          auto suggested_query = _ids_to_queries_map.at(suggested_query_id);
          topk_queries.emplace_back(suggested_query);
        }
        batch_suggestions.emplace_back(topk_queries);
      }

      results.emplace_back(batch_suggestions);
    }

    return results;
  }

  /**
   * @brief Given a vector of queries, returns top k generated queries
   * by the flash instance for each of the query. For instance, if
   * queries is a vector of size n, then the output will be a vector
   * of size n, each of which is also a vector of size k.
   *
   * @param queries
   * @return A vector of suggested queries
   *
   * TODO(blaise): Refactor queryFromList and queryFromFile using a
   *                helper method.
   */
  std::vector<std::vector<std::string>> queryFromList(
      const std::vector<std::string>& queries) {
    std::vector<std::vector<std::string>> outputs;

#pragma omp parallel for default(none) shared(queries, outputs)
    for (const auto& query : queries) {
      auto featurized_query_vector = featurizeSingleQuery(query);

      std::vector<std::vector<uint32_t>> suggested_query_ids =
          _flash_generator->queryBatch(
              /* batch = */ BoltBatch(std::move(featurized_query_vector)),
              /* top_k = */ _flash_generator_config->topk(),
              /* pad_zeros = */ true);

      std::vector<std::string> topk_queries;
      for (auto suggested_query_id : suggested_query_ids[0]) {
        auto suggested_query = _ids_to_queries_map.at(suggested_query_id);
        topk_queries.emplace_back(suggested_query);
      }
      outputs.emplace_back(topk_queries);
    }

    return outputs;
  }

  std::unordered_map<uint32_t, std::string> getIDsToQueryMapping() const {
    return _ids_to_queries_map;
  }

 private:
  /**
   * @brief Constructs a mapping from IDs to correct queries. This allows
   * us to convert the output of queryBatch() into strings representing
   * correct queries.
   *
   * @param file_name: File containing queries (Expected to be a CSV file).
   */
  void buildIDsQueryMapping(const std::string& file_name,
                            bool has_incorrect_queries) {
    try {
      std::ifstream input_file_stream(file_name, std::ios::in);

      uint32_t ids_counter = 0;
      std::string row;

      while (std::getline(input_file_stream, row, '\n')) {
        if (has_incorrect_queries) {
          auto parsed_row = dataset::ProcessorUtils::parseCsvRow(row, ',');
          _ids_to_queries_map[ids_counter] = std::string(parsed_row[1]);
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
    dataset::TextBlockPtr char_trigram_block =
        dataset::CharKGramTextBlock::make(
            /* col = */ 0, /* k = */ 3, /* dim = */ _dimension_for_encodings);
    dataset::TextBlockPtr char_four_gram_block =
        dataset::CharKGramTextBlock::make(
            /* col = */ 0, /* k = */ 4, /* dim = */ _dimension_for_encodings);

    std::vector<std::shared_ptr<dataset::Block>> input_blocks{
        char_trigram_block, char_four_gram_block};

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

  std::unique_ptr<dataset::StreamingGenericDatasetLoader>
  getUnlabeledDatasetLoader(const std::string& file_name,
                            uint8_t correct_query_column_index) {
    auto file_data_loader = dataset::SimpleFileDataLoader::make(
        file_name, _flash_generator_config->batch_size());

    dataset::TextBlockPtr char_trigram_block =
        dataset::CharKGramTextBlock::make(
            /* col = */ correct_query_column_index, /* k = */ 3,
            /* dim = */ _dimension_for_encodings);
    dataset::TextBlockPtr char_four_gram_block =
        dataset::CharKGramTextBlock::make(
            /* col = */ correct_query_column_index, /* k = */ 4,
            /* dim = */ _dimension_for_encodings);

    std::vector<dataset::BlockPtr> input_blocks = {char_trigram_block,
                                                   char_four_gram_block};

    std::vector<dataset::BlockPtr> label_block = {};

    _batch_processor = std::make_shared<dataset::GenericBatchProcessor>(
        input_blocks, label_block);

    return std::make_unique<dataset::StreamingGenericDatasetLoader>(
        file_data_loader, _batch_processor);
  }

  std::shared_ptr<GeneratorConfig> _flash_generator_config;
  std::unique_ptr<Flash<uint32_t>> _flash_generator;

  std::shared_ptr<dataset::GenericBatchProcessor> _batch_processor;

  // Maintains a mapping from the assigned IDs to the original
  // queries loaded from a CSV file. Each unique query in the input
  // CSV file is assigned a unique ID in an ascending order.
  std::unordered_map<uint32_t, std::string> _ids_to_queries_map;

  uint32_t _dimension_for_encodings;

  // private constructor for cereal
  Generator() {}

  friend class cereal::access;
  template <class Archive>
  void serialize(Archive& archive) {
    archive(_flash_generator_config, _flash_generator, _batch_processor,
            _ids_to_queries_map, _dimension_for_encodings);
  }
};  // namespace thirdai::bolt

using GeneratorPtr = std::shared_ptr<Generator>;

}  // namespace thirdai::bolt