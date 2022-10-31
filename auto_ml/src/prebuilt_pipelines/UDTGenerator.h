#pragma once

#include <auto_ml/src/prebuilt_pipelines/UniversalDeepTransformer.h>
#include <generator/src/Generator.h>
#include <search/src/Generator.h>
// #include <search/src/Generator.h>

// using thirdai::bolt::QueryCandidateGeneratorConfig;
// using thirdai::bolt::QueryCandidateGenerator;

namespace thirdai::automl::deployment {

class UDTGenerator : public ModelPipeline, public UniversalDeepTransformerBase {
 public:
  /**
   * Factory method. The arguments below are used to determine what parameters
   * to configure in the QueryCandidateGeneratorConfig.
   * - target_column: Name of the column containing correct queries
   * - source_column: Name of the column containing incorrect queries
   * - dataset_size: Size of the dataset. Options include ["small", "medium",
   * "large"]
   */
  static UniversalDeepTransformerBase buildUDT(
      const std::string& target_column, const std::string& source_column,
      const std::string& dataset_size) {
    (void)target_column;
    (void)source_column;
    (void)dataset_size;

    return;
  }

  void save(const std::string& filename) final { (void)filename; }

  static std::unique_ptr<UDTGenerator> load(const std::string& filename) {
    auto file = filename;
    (void)file;
    return nullptr;
  }

  void trainOnFile(const std::string& filename, bolt::TrainConfig& train_config,
                   std::optional<uint32_t> batch_size_opt,
                   std::optional<uint32_t> max_in_memory_batches) {
    (void)max_in_memory_batches;
    (void)train_config;

    uint32_t batch_size =
        batch_size_opt.value_or(_train_eval_config.defaultBatchSize());

    // trainOnDataLoader(dataset::SimpleFileDataLoader::make(filename,
    // batch_size),
    //                   train_config, max_in_memory_batches);
  }

 private:
  void setupGeneratorConfig() {}

  std::unique_ptr<Generator> _generator;
};

}  // namespace thirdai::automl::deployment