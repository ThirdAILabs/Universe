#pragma once

#include <bolt/python_bindings/CtrlCCheck.h>
#include <bolt/src/nn/model/Model.h>
#include <bolt/src/train/callbacks/Callback.h>
#include <bolt/src/train/metrics/Metric.h>
#include <bolt/src/train/trainer/DistributedComm.h>
#include <bolt/src/train/trainer/Trainer.h>
#include <auto_ml/src/config/ArgumentMap.h>
#include <auto_ml/src/rlhf/RLHFSampler.h>
#include <auto_ml/src/udt/UDTBackend.h>
#include <data/src/ColumnMap.h>
#include <data/src/ColumnMapIterator.h>
#include <data/src/Loader.h>
#include <data/src/TensorConversion.h>
#include <data/src/columns/ArrayColumns.h>
#include <data/src/columns/Column.h>
#include <data/src/columns/ValueColumns.h>
#include <data/src/transformations/AddMachRlhfSamples.h>
#include <data/src/transformations/Pipeline.h>
#include <data/src/transformations/State.h>
#include <data/src/transformations/Transformation.h>
#include <dataset/src/mach/MachIndex.h>
#include <algorithm>
#include <csignal>
#include <cstddef>
#include <cstdint>
#include <iterator>
#include <optional>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

namespace thirdai::automl::udt::utils {

namespace mach = dataset::mach;

class Mach {
 public:
  Mach(uint32_t input_dim, uint32_t num_buckets,
       const config::ArgumentMap& args,
       const std::optional<std::string>& model_config, bool use_sigmoid_bce,
       uint32_t num_hashes, float mach_sampling_threshold,
       bool freeze_hash_tables, std::string input_indices_column,
       std::string input_values_column, std::string label_column,
       std::string bucket_column);

  static auto make(uint32_t input_dim, uint32_t num_buckets,
                   const config::ArgumentMap& args,
                   const std::optional<std::string>& model_config,
                   bool use_sigmoid_bce, uint32_t num_hashes,
                   float mach_sampling_threshold, bool freeze_hash_tables,
                   std::string input_indices_column,
                   std::string input_values_column, std::string label_column,
                   std::string bucket_column) {
    return std::make_shared<Mach>(
        input_dim, num_buckets, args, model_config, use_sigmoid_bce, num_hashes,
        mach_sampling_threshold, freeze_hash_tables,
        std::move(input_indices_column), std::move(input_values_column),
        std::move(label_column), std::move(bucket_column));
  }

  // TODO(Geordie): Rename `columns` to something more descriptive

  void randomlyIntroduceEntities(const data::ColumnMap& columns);

  void introduceEntities(const data::ColumnMap& columns,
                         std::optional<uint32_t> num_buckets_to_sample_opt,
                         uint32_t num_random_hashes);

  void eraseEntity(uint32_t entity);

  void eraseAllEntities();

  bolt::metrics::History train(
      data::ColumnMapIteratorPtr train_iter,
      data::ColumnMapIteratorPtr val_iter, float learning_rate, uint32_t epochs,
      const bolt::metrics::InputMetrics& train_metrics,
      const bolt::metrics::InputMetrics& val_metrics,
      const std::vector<bolt::callbacks::CallbackPtr>& callbacks,
      TrainOptions options, const bolt::DistributedCommPtr& comm);

  void train(data::ColumnMap columns, float learning_rate);

  void trainBuckets(data::ColumnMap columns, float learning_rate);

  bolt::metrics::History evaluate(data::ColumnMapIteratorPtr eval_iter,
                                  const bolt::metrics::InputMetrics& metrics,
                                  bool sparse_inference, bool verbose);

  std::vector<std::vector<std::pair<uint32_t, double>>> predict(
      const data::ColumnMap& columns, bool sparse_inference, uint32_t top_k,
      uint32_t num_scanned_buckets);

  std::vector<std::vector<uint32_t>> predictBuckets(
      const data::ColumnMap& columns, bool sparse_inference,
      std::optional<uint32_t> top_k, bool force_non_empty);

  void upvote(data::ColumnMap upvotes, float learning_rate, uint32_t repeats,
              uint32_t num_balancers, uint32_t epochs, size_t batch_size);

  void associate(data::ColumnMap from_table, const data::ColumnMap& to_table,
                 float learning_rate, uint32_t repeats, uint32_t num_balancers,
                 uint32_t num_buckets, uint32_t epochs, size_t batch_size);

  bolt::metrics::History associateTrain(
      data::ColumnMap from_table, const data::ColumnMap& to_table,
      data::ColumnMap train_data, float learning_rate, uint32_t repeats,
      uint32_t num_buckets, uint32_t epochs, size_t batch_size,
      const bolt::metrics::InputMetrics& metrics, TrainOptions options);

  std::vector<std::vector<std::pair<uint32_t, double>>> score(
      const data::ColumnMap& columns,
      std::vector<std::unordered_set<uint32_t>>& entities,
      std::optional<uint32_t> top_k);

  std::vector<uint32_t> outputCorrectness(const data::ColumnMap& columns,
                                          const std::vector<uint32_t>& labels,
                                          std::optional<uint32_t> num_hashes,
                                          bool sparse_inference);

  bolt::TensorPtr embedding(const data::ColumnMap& columns);

  std::vector<float> entityEmbedding(uint32_t entity) const;

  bolt::ModelPtr& model() { return _model; }

  const mach::MachIndexPtr& index() { return _state->machIndex(); }

  const dataset::mach::MachIndexPtr& index() const {
    return _state->machIndex();
  }

  size_t size() const { return index()->numEntities(); }

  void setIndex(dataset::mach::MachIndexPtr new_index) {
    _state->setMachIndex(std::move(new_index));
    updateSamplingStrategy();
  }

  void setMachSamplingThreshold(float threshold) {
    _mach_sampling_threshold = threshold;
    updateSamplingStrategy();
  }

  void enableRlhf(uint32_t num_balancing_docs,
                  uint32_t num_balancing_samples_per_doc);

 private:
  void updateSamplingStrategy();

  std::vector<uint32_t> topHashesForDoc(
      std::vector<TopKActivationsQueue>&& top_k_per_sample,
      uint32_t num_buckets_to_sample, uint32_t num_random_hashes);

  std::optional<data::ColumnMap> balancingColumnMap(uint32_t num_balancers);

  void teach(data::ColumnMap feedback, float learning_rate,
             uint32_t feedback_repetitions, uint32_t num_balancers,
             uint32_t epochs, size_t batch_size);

  data::ColumnMap associateSamples(data::ColumnMap from_columns,
                                   const data::ColumnMap& to_columns,
                                   uint32_t num_buckets);

  data::LoaderPtr loader(data::ColumnMapIteratorPtr iter, bool train,
                         TrainOptions options) {
    auto transformations = data::Pipeline::make({_label_to_buckets});
    if (train && _add_balancing_samples) {
      transformations = transformations->then(_add_balancing_samples);
    }
    return data::Loader::make(
        std::move(iter), transformations, _state, _bolt_input_columns,
        _bolt_label_columns, options.batchSize(), /* shuffle= */ train,
        options.verbose, options.shuffle_config.min_buffer_size,
        options.shuffle_config.seed);
  }

  void addMachLabels(data::ColumnMap& columns) const {
    columns = _label_to_buckets->apply(std::move(columns), *_state);
  }

  void addDummyLabels(data::ColumnMap& columns) const {
    auto doc_ids = thirdai::data::ValueColumn<uint32_t>::make(
        std::vector<uint32_t>(columns.numRows(), 0),
        std::numeric_limits<uint32_t>::max());
    columns.setColumn(labelColumn(), doc_ids);
  }

  void assertRlhfEnabled() const {
    if (!_state->hasRlhfSampler()) {
      throw std::runtime_error(
          "This model was not configured to support rlhf. Please pass {'rlhf': "
          "True} in the model options or call enable_rlhf().");
    }
  }

  void addRlhfSamplesIfNeeded(const data::ColumnMap& columns) const {
    if (_add_balancing_samples) {
      _add_balancing_samples->apply(columns, *_state);
    }
  }

  bolt::TensorList inputTensors(const data::ColumnMap& columns) const {
    return data::toTensors(columns, _bolt_input_columns);
  }

  bolt::TensorList labelTensors(const data::ColumnMap& columns) const {
    return data::toTensors(columns, _bolt_label_columns);
  }

  const std::string& inputIndicesColumn() const {
    return _bolt_input_columns.front().indices();
  }

  const std::string& inputValuesColumn() const {
    return _bolt_input_columns.front().values().value();
  }

  const std::string& bucketColumn() const {
    return _bolt_label_columns.front().indices();
  }

  const std::string& labelColumn() const {
    return _bolt_label_columns.back().indices();
  }

  size_t inputDim() const { return _model->inputDims().front(); }

  size_t numBuckets() const { return _state->machIndex()->numBuckets(); }

  Mach() {}  // for cereal

  bolt::ModelPtr _model;
  bolt::ComputationPtr _emb;
  float _mach_sampling_threshold;
  bool _freeze_hash_tables;

  data::StatePtr _state;

  data::TransformationPtr _label_to_buckets;
  data::TransformationPtr _add_balancing_samples;
  data::OutputColumnsList _bolt_input_columns;
  data::OutputColumnsList _bolt_label_columns;
  std::vector<std::string> _all_bolt_columns;

  std::mt19937 _mt{341};

  friend class cereal::access;
  template <typename Archive>
  void serialize(Archive& archive);
};

using MachPtr = std::shared_ptr<Mach>;

}  // namespace thirdai::automl::udt::utils