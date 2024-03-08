#pragma once

#include <bolt/src/nn/loss/BinaryCrossEntropy.h>
#include <bolt/src/nn/loss/CategoricalCrossEntropy.h>
#include <bolt/src/nn/model/Model.h>
#include <bolt/src/nn/ops/Embedding.h>
#include <bolt/src/nn/ops/FullyConnected.h>
#include <bolt/src/nn/ops/Input.h>
#include <bolt/src/nn/ops/LayerNorm.h>
#include <bolt/src/nn/tensor/Tensor.h>
#include <bolt/src/train/callbacks/Callback.h>
#include <bolt/src/train/metrics/Metric.h>
#include <bolt/src/train/trainer/DistributedComm.h>
#include <bolt_vector/src/BoltVector.h>
#include <auto_ml/src/featurization/DataTypes.h>
#include <data/src/ColumnMap.h>
#include <data/src/ColumnMapIterator.h>
#include <data/src/TensorConversion.h>
#include <data/src/columns/ValueColumns.h>
#include <data/src/transformations/AddMachMemorySamples.h>
#include <data/src/transformations/MachLabel.h>
#include <data/src/transformations/MachMemory.h>
#include <data/src/transformations/State.h>
#include <data/src/transformations/StringConcat.h>
#include <data/src/transformations/TextTokenizer.h>
#include <data/src/transformations/Transformation.h>
#include <data/src/transformations/cold_start/ColdStartText.h>
#include <data/src/transformations/cold_start/VariableLengthColdStart.h>
#include <dataset/src/blocks/text/TextEncoder.h>
#include <dataset/src/blocks/text/TextTokenizer.h>
#include <dataset/src/dataset_loaders/DatasetLoader.h>
#include <dataset/src/mach/MachIndex.h>
#include <sys/types.h>
#include <utils/text/StringManipulation.h>
#include <algorithm>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

namespace thirdai::automl::mach {

using IdScores = std::vector<std::pair<uint32_t, double>>;

struct TrainOptions {
  uint32_t batch_size;
  std::optional<uint32_t> max_in_memory_batches;
  std::optional<uint32_t> steps_per_validation;
  std::optional<uint32_t> logging_interval;
  bool sparse_validation;
  bool verbose;
  bool freeze_hash_tables;
  dataset::DatasetShuffleConfig shuffle_config;
};

static constexpr const char* bucket_column = "__buckets__";
static constexpr const char* input_indices_column = "__input_indices__";
static constexpr const char* input_values_column = "__input_values__";
static constexpr const char* label_indices_column = "__label_indices__";
static constexpr const char* label_values_column = "__label_values__";

bolt::ModelPtr defaultModel(uint32_t text_feature_dim, uint32_t embedding_dim,
                            uint32_t output_dim, bool embedding_bias,
                            bool output_bias, bool normalize_embeddings,
                            const std::string& embedding_act_func,
                            const std::string& output_act_func);

data::ValueFillType toValueFillType(const std::string& output_act_func);

class MachRetriever {
 public:
  MachRetriever(std::string text_column, const std::string& id_column,
                uint32_t num_hashes, uint32_t output_dim,
                uint32_t embedding_dim, uint32_t text_feature_dim,
                uint32_t output_bias, uint32_t embedding_bias,
                bool normalize_embeddings, const std::string& output_act_func,
                const std::string& embedding_act_func,
                const std::string& tokenizer,
                const std::string& contextual_encoding, bool lowercase,
                float mach_sampling_threshold, uint32_t num_buckets_to_eval,
                size_t memory_max_ids, size_t memory_max_samples_per_id);

  void introduce(const data::ColumnMapIteratorPtr& iter,
                 const std::vector<std::string>& strong_column_names,
                 const std::vector<std::string>& weak_column_names,
                 bool phrase_sampling,
                 std::optional<uint32_t> num_buckets_to_sample_opt,
                 uint32_t num_random_hashes, bool load_balancing,
                 bool sort_random_hashes);

  void introduce(data::ColumnMap columns,
                 const std::vector<std::string>& strong_column_names,
                 const std::vector<std::string>& weak_column_names,
                 bool phrase_sampling,
                 std::optional<uint32_t> num_buckets_to_sample_opt,
                 uint32_t num_random_hashes, bool load_balancing,
                 bool sort_random_hashes);

  dataset::mach::MachIndexPtr index() { return _state->machIndex(); }

  void erase(const std::vector<uint32_t>& ids) {
    for (uint32_t id : ids) {
      index()->erase(id);
    }
  };

  void coldstart(data::ColumnMapIteratorPtr iter,
                 const std::vector<std::string>& strong_column_names,
                 const std::vector<std::string>& weak_column_names,
                 std::optional<data::VariableLengthConfig> variable_length,
                 float learning_rate, uint32_t epochs,
                 const std::vector<std::string>& train_metrics,
                 data::ColumnMapIteratorPtr val_iter,
                 const std::vector<std::string>& val_metrics,
                 const TrainOptions& options,
                 const std::vector<bolt::callbacks::CallbackPtr>& callbacks,
                 const bolt::DistributedCommPtr& comm);

  void train(data::ColumnMapIteratorPtr iter, float learning_rate,
             uint32_t epochs, const std::vector<std::string>& train_metrics,
             data::ColumnMapIteratorPtr val_iter,
             const std::vector<std::string>& val_metrics,
             const TrainOptions& options,
             const std::vector<bolt::callbacks::CallbackPtr>& callbacks,
             const bolt::DistributedCommPtr& comm);

  std::vector<IdScores> search(data::ColumnMap queries, uint32_t top_k,
                               bool sparse_inference);

  std::vector<IdScores> rank(
      data::ColumnMap queries,
      const std::vector<std::unordered_set<uint32_t>>& choices,
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

 private:
  bolt::TensorList inputTensors(const data::ColumnMap& columns) {
    return data::toTensors(columns, _bolt_input_columns);
  }

  bolt::TensorList labelTensors(const data::ColumnMap& columns) {
    return data::toTensors(columns, _bolt_label_columns);
  }

  data::TransformationPtr phraseSampling(
      const std::vector<std::string>& strong_column_names,
      const std::vector<std::string>& weak_column_names,
      std::optional<data::VariableLengthConfig> variable_length) {
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
      if (seen.count(id) || index()->has(id)) {
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
  data::TransformationPtr _id_transform;
  data::TransformationPtr _add_mach_memory_samples;

  data::OutputColumnsList _bolt_input_columns;
  data::OutputColumnsList _bolt_label_columns;
  std::vector<std::string> _all_bolt_columns;

  float _mach_sampling_threshold;
  uint32_t _num_buckets_to_eval;
};

}  // namespace thirdai::automl::mach