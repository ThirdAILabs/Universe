#pragma once

#include <cereal/access.hpp>
#include <cereal/types/memory.hpp>
#include <cereal/types/polymorphic.hpp>
#include <auto_ml/src/prebuilt_pipelines/UniversalDeepTransformer.h>
#include <auto_ml/src/prebuilt_pipelines/UniversalDeepTransformerBase.h>
#include <search/src/Generator.h>

namespace thirdai::automl::deployment {

using thirdai::bolt::QueryCandidateGenerator;
using thirdai::bolt::QueryCandidateGeneratorConfig;

class UDTQueryCandidateGenerator : public UniversalDeepTransformerBase {
  static inline const std::string DEFAULT_HASH_FUNCTION = "minhash";
  static inline const uint32_t DEFAULT_NUM_TABLES = 128;
  static inline const uint32_t DEFAULT_HASHES_PER_TABLE = 5;
  static inline const uint32_t DEFAULT_HASH_TABLE_RANGE = 100000;
  static inline const std::vector<uint32_t> DEFAULT_N_GRAMS = {3, 4};

 public:
  explicit UDTQueryCandidateGenerator(QueryCandidateGenerator&& model)
      : _generator(
            std::make_shared<QueryCandidateGenerator>(std::move(model))) {}

  /**
   * Factory method for the query reformulation UDT model. The arguments below
   * are used to determine what parameters to configure in the
   * QueryCandidateGeneratorConfig.
   * - target_column: Name of the column containing correct queries
   * - source_column: Name of the column containing incorrect queries
   * - dataset_size: Size of the dataset. Options include ["small", "medium",
   * "large"]
   */
  static UniversalDeepTransformerBasePtr buildUDT(
      const uint32_t& target_column_index, const uint32_t& source_column_index,
      const std::string& dataset_size) {
    (void)dataset_size;

    auto generator_config = QueryCandidateGeneratorConfig(
        /* hash_function = */ DEFAULT_HASH_FUNCTION,
        /* num_tables = */ DEFAULT_NUM_TABLES,
        /* hashes_per_table = */ DEFAULT_HASHES_PER_TABLE,
        /* range = */ DEFAULT_HASH_TABLE_RANGE,
        /* n_grams = */ DEFAULT_N_GRAMS,
        /* reservoir_size */ std::nullopt,
        /* source_column_index = */ source_column_index,
        /* target_column_index = */ target_column_index,
        /* has_incorrect_queries = */ true);

    auto generator = QueryCandidateGenerator::make(
        std::make_shared<QueryCandidateGeneratorConfig>(generator_config));

    return std::make_unique<UDTQueryCandidateGenerator>(
        /* model = */ std::move(generator));
  }

  void save(const std::string& filename) final {
    std::ofstream file_stream =
        dataset::SafeFileIO::ofstream(filename, std::ios::binary);
    cereal::BinaryOutputArchive oarchive(file_stream);

    oarchive(*this);
  }

  static std::shared_ptr<UniversalDeepTransformerBase> load(
      const std::string& filename) {
    std::ifstream file_stream =
        dataset::SafeFileIO::ifstream(filename, std::ios::binary);

    cereal::BinaryInputArchive iarchive(file_stream);

    std::shared_ptr<UDTQueryCandidateGenerator> deserialized_into(
        new UDTQueryCandidateGenerator());

    iarchive(*deserialized_into);

    return deserialized_into;
  }

  std::shared_ptr<QueryCandidateGenerator> generator() const {
    return _generator;
  }

 private:
  // Private constructor for cereal
  UDTQueryCandidateGenerator() {}

  std::shared_ptr<QueryCandidateGenerator> _generator;

  friend class cereal::access;
  template <class Archive>
  void serialize(Archive& archive) {
    archive(cereal::base_class<UniversalDeepTransformerBase>(this), _generator);
  }
};

using UDTQueryCandidateGeneratorPtr =
    std::unique_ptr<UDTQueryCandidateGenerator>;

}  // namespace thirdai::automl::deployment

CEREAL_REGISTER_TYPE(thirdai::automl::deployment::UDTQueryCandidateGenerator)