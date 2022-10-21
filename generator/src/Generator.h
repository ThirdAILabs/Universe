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
    std::stringstream output;
    cereal::BinaryOutputArchive output_archive(output);

    output_archive(*this);

    std::ofstream filestream =
        dataset::SafeFileIO::ofstream(config_file_name, std::ios::binary);

    std::string string_representation = output.str();
    filestream.write(string_representation.data(),
                     string_representation.size());
  }

  static std::shared_ptr<GeneratorConfig> load(
      const std::string& config_file_name) {
    std::ifstream filestream =
        dataset::SafeFileIO::ifstream(config_file_name, std::ios::binary);

    std::stringstream buffer;
    buffer << filestream.rdbuf();

    cereal::BinaryInputArchive input_archive(buffer);
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
   * @brief Builds a Flash index
   *
   * @param file_name
   * @return std::shared_ptr<Generator>
   *
   * TODO(blaise): Combine buildFlashGenerator and buildFlashGeneratorPair into
   * one function since they are pretty much indentical
   */
  std::shared_ptr<Generator> buildFlashGenerator(const std::string& file_name) {
    buildIDsQueryMapping(file_name);

    auto* data = loadDataInMemory(/* file_name = */ file_name,
                                  /* correct_query_column_index = */ 0)
                     .get();
    _flash_generator = std::make_unique<Flash<uint32_t>>(
        _flash_generator_config->getHashFunction());
    _flash_generator->addDataset(*data);

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
   * @return std::shared_ptr<Generator>
   */
  std::shared_ptr<Generator> buildFlashGeneratorFromQueryPairs(
      const std::string& file_name) {
    auto data = loadDataInMemory(/* file_name = */ file_name,
                                 /* correct_query_column_index = */ 1);
    _flash_generator = std::make_unique<Flash<uint32_t>>(
        _flash_generator_config->getHashFunction());
    _flash_generator->addDataset(*data);

    return shared_from_this();
  }

  std::vector<std::vector<std::vector<uint32_t>>> queryFromFile(
      const std::string& query_file) {
    auto query_data = loadDataInMemory(/* file_name = */ query_file,
                                       /* correct_query_column_index = */ 0);

    std::vector<std::vector<std::vector<uint32_t>>> results;

    for (uint32_t batch_index = 0; batch_index < query_data->len();
         batch_index++) {
      auto query_result = _flash_generator->queryBatch(
          /* batch = */ query_data->at(batch_index),
          /* top_k = */ _flash_generator_config->topk(),
          /* pad_zeros = */ true);

      results.push_back(query_result);
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
   */
  std::vector<std::vector<std::string>> queryFromList(
      const std::vector<std::string>& queries) {
    std::vector<std::vector<std::string>> outputs;

#pragma omp parallel for default(none) shared(queries, outputs)
    {
      for (const auto& query : queries) {
        auto featurized_query_vector = featurizeSingleQuery(query);

        std::vector<std::vector<uint32_t>> suggested_query_ids =
            _flash_generator->queryBatch(
                /* batch = */ BoltBatch(std::move(featurized_query_vector)),
                /* top_k = */ _flash_generator_config->topk(),
                /* pad_zeros = */ true);

        std::vector<std::string> topk_queries;
        for (auto suggested_query_id : suggested_query_ids[0]) {
          if (suggested_query_id == 0) {
            topk_queries.emplace_back("");
            continue;
          }
          auto suggested_query = _ids_to_queries_map.at(suggested_query_id);
          topk_queries.emplace_back(suggested_query);
        }
        outputs.emplace_back(topk_queries);
      }
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
  void buildIDsQueryMapping(const std::string& file_name) {
    try {
      std::ifstream input_file_stream(file_name, std::ios::in);

      // The ID 0 is reserved for empty string.
      uint32_t ids_counter = 1;
      std::string row;

      while (std::getline(input_file_stream, row, '\n')) {
        _ids_to_queries_map[ids_counter] = row;
        ids_counter += 1;
      }
      input_file_stream.close();

    } catch (const std::ifstream::failure& exception) {
      throw std::invalid_argument("Invalid input file name.");
    }
  }

  dataset::BoltDatasetPtr loadDataInMemory(
      const std::string& file_name, uint32_t correct_query_column_index) const {
    dataset::TextBlockPtr char_trigram_block =
        dataset::CharKGramTextBlock::make(
            /* col = */ correct_query_column_index, /* k = */ 3,
            /* dim = */ _dimension_for_encodings);

    dataset::TextBlockPtr char_four_gram_block =
        dataset::CharKGramTextBlock::make(
            /* col = */ correct_query_column_index, /* k = */ 4,
            /* dim = */ _dimension_for_encodings);

    std::vector<std::shared_ptr<dataset::Block>> input_blocks{
        char_trigram_block, char_four_gram_block};

    auto data_loader = dataset::StreamingGenericDatasetLoader(
        /* filename = */ file_name, /* input_blocks = */ input_blocks,
        /* label_blocks = */ {},
        /* batch_size = */ _flash_generator_config->batch_size());

    auto [data, _] = data_loader.loadInMemory();

    return data;
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

  std::shared_ptr<GeneratorConfig> _flash_generator_config;
  std::unique_ptr<Flash<uint32_t>> _flash_generator;

  // Maintains a mapping from the assigned IDs to the original
  // queries loaded from a CSV file. Each unique query in the input
  // CSV file is assigned a unique ID in an ascending order.
  // This field is optional until the query CSV file has been loaded
  // in memory.
  std::unordered_map<uint32_t, std::string> _ids_to_queries_map;

  uint32_t _dimension_for_encodings;

  // private constructor for cereal
  Generator() {}

  friend class cereal::access;
  template <class Archive>
  void serialize(Archive& archive) {
    archive(_flash_generator_config, _flash_generator, _ids_to_queries_map,
            _dimension_for_encodings);
  }
};  // namespace thirdai::bolt

using GeneratorPtr = std::shared_ptr<Generator>;

}  // namespace thirdai::bolt