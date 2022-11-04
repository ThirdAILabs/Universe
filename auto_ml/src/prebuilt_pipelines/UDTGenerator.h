#pragma once

#include <auto_ml/src/prebuilt_pipelines/UniversalDeepTransformer.h>
#include <search/src/Generator.h>

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
    (void)target_column_index;
    (void)source_column_index;
    (void)dataset_size;

    auto generator_config = QueryCandidateGeneratorConfig(
        /* hash_function = */ DEFAULT_HASH_FUNCTION,
        /* num_tables = */ DEFAULT_NUM_TABLES,
        /* hashes_per_table = */ DEFAULT_HASHES_PER_TABLE, /* top_k = */ 5,
        /* n_grams = */ DEFAULT_N_GRAMS,
        /* has_incorrect_queries = */ true, /* input_dim = */ 100);

    auto generator = QueryCandidateGenerator::make(
        std::make_shared<QueryCandidateGeneratorConfig>(generator_config));

    return UDTGenerator(/* model = */ std::move(generator));
  }

  static void save(const std::string& filename) { (void)filename; }

  static std::unique_ptr<UDTGenerator> load(const std::string& filename) {
    auto file = filename;
    (void)file;
    return nullptr;
  }

  static void trainOnFile(const std::string& filename,
                          bolt::TrainConfig& train_config,
                          std::optional<uint32_t> batch_size_opt,
                          std::optional<uint32_t> max_in_memory_batches) {
    (void)max_in_memory_batches;
    (void)train_config;
    (void)filename;
    (void)batch_size_opt;

    // uint32_t batch_size =
    //     batch_size_opt.value_or(_train_eval_config.defaultBatchSize());

    // trainOnDataLoader(dataset::SimpleFileDataLoader::make(filename,
    // batch_size),
    //                   train_config, max_in_memory_batches);
  }

 private:
  explicit UDTGenerator(QueryCandidateGenerator&& model)
      : QueryCandidateGenerator(std::move(model)) {}


  std::unique_ptr<QueryCandidateGenerator> _generator;
};

}  // namespace thirdai::automl::deployment