#pragma once

#include <auto_ml/src/prebuilt_pipelines/UniversalDeepTransformer.h>
#include <auto_ml/src/prebuilt_pipelines/UniversalDeepTransformerBase.h>
#include <search/src/Generator.h>
#include <cereal/access.hpp>
#include <cereal/types/polymorphic.hpp>

namespace thirdai::automl::deployment {

using thirdai::bolt::QueryCandidateGenerator;
using thirdai::bolt::QueryCandidateGeneratorConfig;

class UDTGenerator : public QueryCandidateGenerator,
                     public UniversalDeepTransformerBase {
  static inline const std::string DEFAULT_HASH_FUNCTION = "minhash";
  static inline const uint32_t DEFAULT_NUM_TABLES = 128;
  static inline const uint32_t DEFAULT_HASHES_PER_TABLE = 5;
  static inline const std::vector<uint32_t> DEFAULT_N_GRAMS = {3, 4};

 public:
  /**
   * Factory method. The arguments below are used to determine what parameters
   * to configure in the QueryCandidateGeneratorConfig.
   * - target_column: Name of the column containing correct queries
   * - source_column: Name of the column containing incorrect queries
   * - dataset_size: Size of the dataset. Options include ["small", "medium",
   * "large"]
   */
  static UDTGenerator buildUDT(const uint32_t& target_column_index,
                               const uint32_t& source_column_index,
                               const std::string& dataset_size) {
    (void)dataset_size;

    auto generator_config = QueryCandidateGeneratorConfig(
        /* hash_function = */ DEFAULT_HASH_FUNCTION,
        /* num_tables = */ DEFAULT_NUM_TABLES,
        /* hashes_per_table = */ DEFAULT_HASHES_PER_TABLE,
        /* range = */ 100000,
        /* n_grams = */ DEFAULT_N_GRAMS,
        /* has_incorrect_queries = */ true,
        /* source_column_index = */ source_column_index,
        /* target_column_index = */ target_column_index);

    auto generator = QueryCandidateGenerator::make(
        std::make_shared<QueryCandidateGeneratorConfig>(generator_config));

    return UDTGenerator(/* model = */ std::move(generator));
  }

  void save(const std::string& filename) override { 
    std::ofstream file_stream = dataset::SafeFileIO::ofstream(filename, std::ios::binary);
    cereal::BinaryOutputArchive oarchive(file_stream);

    oarchive(*this);
  }

  static std::unique_ptr<UDTGenerator> load(const std::string& filename) final {
    std::ifstream file_stream = dataset::SafeFileIO::ifstream(filename, std::ios::binary);

    cereal::BinaryInputArchive iarchive(file_stream);
    std::unique_ptr<UDTGenerator> deserialized_into(new UDTGenerator());
    iarchive(*deserialized_into);

    return deserialized_into;
  }

 private:
  explicit UDTGenerator(QueryCandidateGenerator&& model)
      : QueryCandidateGenerator(std::move(model)) {}

  // Private constructor for cereal
  UDTGenerator() {}

  friend class cereal::access;
  template <class Archive>
  void serialize(Archive& archive) {
    archive(cereal::base_class<UniversalDeepTransformerBase>(this));
  }


  std::unique_ptr<QueryCandidateGenerator> _generator;
};

}  // namespace thirdai::automl::deployment

CEREAL_REGISTER_TYPE(thirdai::automl::deployment::UDTGenerator)