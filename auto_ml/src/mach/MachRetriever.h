#pragma once

#include <bolt/src/nn/model/Model.h>
#include <bolt/src/train/callbacks/Callback.h>
#include <bolt/src/train/metrics/Metric.h>
#include <bolt/src/train/trainer/Trainer.h>
#include <archive/src/Archive.h>
#include <auto_ml/src/mach/MachConfig.h>
#include <data/src/ColumnMap.h>
#include <data/src/ColumnMapIterator.h>
#include <data/src/TensorConversion.h>
#include <data/src/columns/ValueColumns.h>
#include <data/src/transformations/AddMachMemorySamples.h>
#include <data/src/transformations/MachLabel.h>
#include <data/src/transformations/MachMemory.h>
#include <data/src/transformations/SpladeAugmentation.h>
#include <data/src/transformations/State.h>
#include <data/src/transformations/StringConcat.h>
#include <data/src/transformations/TextTokenizer.h>
#include <data/src/transformations/Transformation.h>
#include <data/src/transformations/cold_start/ColdStartText.h>
#include <data/src/transformations/cold_start/VariableLengthColdStart.h>
#include <dataset/src/dataset_loaders/DatasetLoader.h>
#include <dataset/src/mach/MachIndex.h>
#include <memory>
#include <utility>

namespace thirdai::automl::mach {

using IdScores = std::vector<std::pair<uint32_t, double>>;

struct TrainOptions {
  size_t batch_size = 2048;
  std::optional<size_t> max_in_memory_batches = std::nullopt;
  bool verbose = true;
  bolt::InterruptCheck interrupt_check = std::nullopt;
};

struct ColdStartOptions : public TrainOptions {
  std::optional<data::VariableLengthConfig> variable_length =
      data::VariableLengthConfig();
  std::optional<data::SpladeConfig> splade_config = std::nullopt;
};

class MachRetriever {
 public:
  explicit MachRetriever(const MachConfig& config);

  explicit MachRetriever(const ar::Archive& archive);

  void introduce(const data::ColumnMapIteratorPtr& data,
                 const std::vector<std::string>& strong_column_names,
                 const std::vector<std::string>& weak_column_names,
                 bool text_augmentation,
                 std::optional<uint32_t> num_buckets_to_sample_opt,
                 uint32_t num_random_hashes, bool load_balancing,
                 bool sort_random_hashes);

  void introduce(data::ColumnMap data,
                 const std::vector<std::string>& strong_column_names,
                 const std::vector<std::string>& weak_column_names,
                 bool text_augmentation,
                 std::optional<uint32_t> num_buckets_to_sample_opt,
                 uint32_t num_random_hashes, bool load_balancing,
                 bool sort_random_hashes);

  bolt::metrics::History coldstart(
      const data::ColumnMapIteratorPtr& data,
      const std::vector<std::string>& strong_cols,
      const std::vector<std::string>& weak_cols, float learning_rate,
      uint32_t epochs, const std::vector<std::string>& metrics,
      const std::vector<bolt::callbacks::CallbackPtr>& callbacks,
      const ColdStartOptions& options = ColdStartOptions());

  bolt::metrics::History train(
      const data::ColumnMapIteratorPtr& data, float learning_rate,
      uint32_t epochs, const std::vector<std::string>& metrics,
      const std::vector<bolt::callbacks::CallbackPtr>& callbacks,
      const TrainOptions& options = TrainOptions());

  bolt::metrics::History evaluate(const data::ColumnMapIteratorPtr& data,
                                  const std::vector<std::string>& metrics,
                                  bool verbose);

  std::vector<IdScores> search(data::ColumnMap queries, uint32_t top_k,
                               bool sparse_inference);

  std::vector<IdScores> rank(
      data::ColumnMap queries,
      const std::vector<std::unordered_set<uint32_t>>& candidates,
      std::optional<uint32_t> top_k, bool sparse_inference);

  std::vector<std::vector<uint32_t>> predictBuckets(
      const data::ColumnMap& columns, bool sparse_inference,
      std::optional<uint32_t> top_k, bool force_non_empty);

  void upvote(data::ColumnMap upvotes, uint32_t num_upvote_samples,
              uint32_t num_balancing_samples, float learning_rate,
              uint32_t epochs, size_t batch_size);

  void associate(data::ColumnMap from_columns,
                 const data::ColumnMap& to_columns, uint32_t num_buckets,
                 uint32_t num_association_samples,
                 uint32_t num_balancing_samples, float learning_rate,
                 uint32_t epochs, bool force_non_empty, size_t batch_size);

  dataset::mach::MachIndexPtr index() { return _state->machIndex(); }

  void erase(const std::vector<uint32_t>& ids) {
    for (uint32_t id : ids) {
      index()->erase(id);
    }
  }

  ar::ConstArchivePtr toArchive(bool with_optimizer) const;

  static std::shared_ptr<MachRetriever> fromArchive(const ar::Archive& archive);

 private:
  bolt::TensorList inputTensors(const data::ColumnMap& columns) {
    return data::toTensors(columns, _bolt_input_columns);
  }

  bolt::TensorList labelTensors(const data::ColumnMap& columns) {
    return data::toTensors(columns, _bolt_label_columns);
  }

  data::TransformationPtr textAugmentation(
      const std::vector<std::string>& strong_column_names,
      const std::vector<std::string>& weak_column_names,
      std::optional<data::VariableLengthConfig> variable_length,
      const std::optional<data::SpladeConfig>& splade_config) {
    if (splade_config) {
      throw std::invalid_argument("Splade is not yet implemented.");
    }
    if (variable_length) {
      return std::make_shared<data::VariableLengthColdStart>(
          /* strong_column_names= */ strong_column_names,
          /* weak_column_names= */ weak_column_names,
          /* output_column_name= */ _text_column,
          /* config= */ *variable_length);
    }

    return std::make_shared<data::ColdStartTextAugmentation>(
        /* strong_column_names= */ strong_column_names,
        /* weak_column_names= */ weak_column_names,
        /* output_column_name= */ _text_column);
  }

  data::TransformationPtr textConcat(
      const std::vector<std::string>& strong_column_names,
      const std::vector<std::string>& weak_column_names) {
    std::vector<std::string> all_columns = weak_column_names;
    all_columns.insert(all_columns.end(), strong_column_names.begin(),
                       strong_column_names.end());
    return std::make_shared<data::StringConcat>(all_columns, _text_column);
  }

  std::vector<uint32_t> topHashesForDoc(
      std::vector<std::vector<ValueIndexPair>>&& top_k_per_sample,
      uint32_t num_buckets_to_sample, uint32_t approx_num_hashes_per_bucket,
      uint32_t num_random_hashes, bool load_balancing,
      bool sort_random_hashes) const;

  void updateSamplingStrategy();

  void assertUniqueIds(const data::ColumnMap& columns) {
    std::unordered_set<uint32_t> seen;
    auto ids = columns.getValueColumn<uint32_t>(_id_column);
    for (uint32_t i = 0; i < ids->numRows(); i++) {
      uint32_t id = ids->value(i);
      if (seen.count(id) || index()->contains(id)) {
        throw std::invalid_argument("Found duplicate ID " + std::to_string(id));
      }
      seen.insert(id);
    }
  }

  bolt::metrics::InputMetrics getMetrics(
      const std::vector<std::string>& metric_names, const std::string& prefix);

  void teach(data::ColumnMap feedback, float learning_rate,
             uint32_t feedback_repetitions, uint32_t num_balancers,
             uint32_t epochs, size_t batch_size);

  data::StatePtr _state;
  bolt::ModelPtr _model;

  std::string _text_column;
  std::string _id_column;

  // TODO(Geordie): Rename things
  data::TransformationPtr _text_transform;
  data::TransformationPtr _map_to_buckets;
  data::TransformationPtr _add_mach_memory_samples;

  data::OutputColumnsList _bolt_input_columns;
  data::OutputColumnsList _bolt_label_columns;
  std::vector<std::string> _all_bolt_columns;

  float _mach_sampling_threshold;
  uint32_t _n_buckets_to_eval;
  std::optional<uint32_t> _freeze_tables_epoch;
};

using MachRetrieverPtr = std::shared_ptr<MachRetriever>;

}  // namespace thirdai::automl::mach