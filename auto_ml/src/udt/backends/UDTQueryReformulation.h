#pragma once

#include <auto_ml/src/config/ArgumentMap.h>
#include <auto_ml/src/udt/Defaults.h>
#include <auto_ml/src/udt/UDTBackend.h>
#include <auto_ml/src/udt/symspell.h>
#include <dataset/src/DataSource.h>
#include <dataset/src/Datasets.h>
#include <dataset/src/blocks/BlockInterface.h>
#include <dataset/src/dataset_loaders/DatasetLoader.h>
#include <dataset/src/featurizers/TabularFeaturizer.h>
#include <dataset/src/utils/ThreadSafeVocabulary.h>
#include <search/src/Flash.h>
#include <optional>
#include <unordered_map>

namespace thirdai::automl::udt {

class UDTQueryReformulation final : public UDTBackend {
 public:
  UDTQueryReformulation(std::optional<std::string> incorrect_column_name,
                        std::string correct_column_name,
                        const std::string& dataset_size,
                        std::optional<bool> use_spell_checker, char delimiter,
                        const std::optional<std::string>& model_config,
                        const config::ArgumentMap& user_args);

  py::object train(const dataset::DataSourcePtr& data, float learning_rate,
                   uint32_t epochs,
                   const std::vector<std::string>& train_metrics,
                   const dataset::DataSourcePtr& val_data,
                   const std::vector<std::string>& val_metrics,
                   const std::vector<CallbackPtr>& callbacks,
                   TrainOptions option,
                   const bolt::DistributedCommPtr& comm) final;

  py::object evaluate(const dataset::DataSourcePtr& data,
                      const std::vector<std::string>& metrics,
                      bool sparse_inference, bool verbose,
                      std::optional<uint32_t> top_k) final;

  py::object predict(const MapInput& sample, bool sparse_inference,
                     bool return_predicted_class,
                     std::optional<uint32_t> top_k) final;

  py::object predictBatch(const MapInputBatch& sample, bool sparse_inference,
                          bool return_predicted_class,
                          std::optional<uint32_t> top_k) final;

  //   py::object predictSymSpell(const MapInput& sample, bool sparse_inference,
  //                      bool return_predicted_class,
  //                      std::optional<uint32_t> top_k) final;
 private:
  bool containsColumn(const dataset::DataSourcePtr& data,
                      const std::string& column_name) const;

  std::pair<dataset::BoltDatasetPtr, dataset::BoltDatasetPtr> loadData(
      const dataset::DataSourcePtr& data, const std::string& col_to_hash,
      bool include_labels, uint32_t batch_size, bool verbose);

  std::pair<MapInputBatch, std::vector<uint32_t>> generate_candidates(
      const MapInputBatch& samples);

  std::pair<std::vector<std::string>, std::vector<float>> accumulate_scores(
      std::vector<std::vector<std::string>> phrases,
      std::vector<std::vector<float>> phrase_scores,
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
  static std::unique_ptr<search::Flash<uint32_t>> defaultFlashIndex(
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

  UDTQueryReformulation() {}

  friend class cereal::access;
  template <class Archive>
  void serialize(Archive& archive);

  std::unique_ptr<search::Flash<uint32_t>> _flash_index;

  dataset::TabularFeaturizerPtr _inference_featurizer;

  dataset::ThreadSafeVocabularyPtr _phrase_id_map;

  std::optional<std::string> _incorrect_column_name;
  std::string _correct_column_name;
  std::optional<bool> _use_spell_checker;
  SymPreTrainer _pretrainer;
  std::vector<uint32_t> _n_grams = defaults::N_GRAMS_FOR_GENERATOR;

  char _delimiter;
};

}  // namespace thirdai::automl::udt