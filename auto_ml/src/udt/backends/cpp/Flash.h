#pragma once

#include <archive/src/Archive.h>
#include <auto_ml/src/config/ArgumentMap.h>
#include <auto_ml/src/udt/Defaults.h>
#include <dataset/src/DataSource.h>
#include <dataset/src/Datasets.h>
#include <dataset/src/blocks/BlockInterface.h>
#include <dataset/src/dataset_loaders/DatasetLoader.h>
#include <dataset/src/featurizers/TabularFeaturizer.h>
#include <dataset/src/utils/ThreadSafeVocabulary.h>
#include <search/src/Flash.h>
#include <utils/src/SymSpellBackend/symspell.h>
#include <algorithm>
#include <optional>
#include <string>
#include <unordered_map>

using IdScorePairs = std::pair<std::vector<std::vector<uint32_t>>,
                               std::vector<std::vector<float>>>;

namespace thirdai::automl::udt {

class Flash {
 public:
  Flash(std::optional<std::string> input_column, std::string target_column,
        char delimiter, const std::optional<std::string>& model_config,
        const config::ArgumentMap& user_args);

  static std::unique_ptr<Flash> make(
      const std::optional<std::string>& input_column,
      const std::string& target_column, const std::string& dataset_size);

  explicit Flash(const ar::Archive& archive);

  void train(const dataset::DataSourcePtr& data,
             std::optional<size_t> batch_size, bool verbose);

  std::unordered_map<std::string, std::vector<float>> evaluate(
      const dataset::DataSourcePtr& data, uint32_t top_k, bool verbose);

  std::pair<std::vector<std::vector<string>>, std::vector<std::vector<float>>>
  predictBatch(const MapInputBatch& sample, std::optional<uint32_t> top_k);

  ar::ConstArchivePtr toArchive(bool with_optimizer) const;

  static std::unique_ptr<Flash> fromArchive(const ar::Archive& archive);

  static std::string type() { return "udt_query_reformulation"; }

 private:
  bool containsColumn(const dataset::DataSourcePtr& data,
                      const std::string& column_name) const;

  std::pair<dataset::BoltDatasetPtr, dataset::BoltDatasetPtr> loadData(
      const dataset::DataSourcePtr& data, const std::string& col_to_hash,
      bool include_labels, uint32_t batch_size, bool verbose);

  // Retrieves the top_k accumulated results for generated candidates per query
  // batch. If use_spell_checker is set to true, it utilizes symspell generated
  // candidates and accumulates it's results otherwise, it returns the top_k
  // results for the query batch.

  IdScorePairs queryBatchResults(const MapInputBatch& sample,
                                 std::optional<uint32_t> top_k);

  void addDataToIndex(const dataset::BoltDatasetPtr& data,
                      const dataset::BoltDatasetPtr& labels,
                      std::optional<ProgressBar>& bar,
                      licensing::TrainPermissionsToken token);

  std::vector<std::string> idsToPhrase(const std::vector<uint32_t>& ids);

  std::unordered_map<std::string, double> computeMetrics(
      const std::vector<std::vector<std::string>>& candidates,
      const dataset::BoltDatasetPtr& labels,
      const std::vector<std::string>& metrics);

  // Returns the default flash instance to use for the given dataset size if no
  // model_config is specified.
  static std::unique_ptr<search::Flash> defaultFlashIndex(
      const std::string& dataset_size);

  static dataset::BlockList ngramBlockList(
      const std::string& column_name, const std::vector<uint32_t>& n_grams);

  static uint32_t recall(
      const std::vector<std::vector<uint32_t>>& retrieved_ids,
      const BoltBatch& labels);

  static void requireTopK(const std::optional<uint32_t>& top_k) {
    if (!top_k) {
      throw std::invalid_argument(
          "top_k is a required argument for query reformulation.");
    }
  }

  std::unique_ptr<search::Flash> _flash_index;

  dataset::TabularFeaturizerPtr _inference_featurizer;

  dataset::ThreadSafeVocabularyPtr _phrase_id_map;

  std::optional<std::string> _incorrect_column_name;
  std::string _correct_column_name;
  bool _use_spell_checker;

  SymSpellPtr _symspell_backend;
  std::vector<uint32_t> _n_grams = defaults::N_GRAMS_FOR_GENERATOR;

  char _delimiter;
};

}  // namespace thirdai::automl::udt