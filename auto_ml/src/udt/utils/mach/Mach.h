#pragma once

#include "MachDetails.h"
#include <bolt/src/nn/loss/BinaryCrossEntropy.h>
#include <bolt/src/nn/model/Model.h>
#include <bolt/src/nn/ops/Embedding.h>
#include <bolt/src/nn/ops/FullyConnected.h>
#include <bolt/src/nn/ops/Input.h>
#include <bolt/src/nn/tensor/Tensor.h>
#include <bolt/src/train/callbacks/Callback.h>
#include <bolt/src/train/metrics/Metric.h>
#include <bolt/src/train/trainer/DistributedComm.h>
#include <auto_ml/src/featurization/ReservedColumns.h>
#include <auto_ml/src/udt/UDTBackend.h>
#include <auto_ml/src/udt/utils/Models.h>
#include <data/src/ColumnMap.h>
#include <data/src/ColumnMapIterator.h>
#include <data/src/Loader.h>
#include <data/src/TensorConversion.h>
#include <data/src/columns/ArrayColumns.h>
#include <data/src/columns/ValueColumns.h>
#include <data/src/transformations/MachLabel.h>
#include <data/src/transformations/Pipeline.h>
#include <data/src/transformations/RLHFSampler.h>
#include <data/src/transformations/State.h>
#include <data/src/transformations/Transformation.h>
#include <dataset/src/mach/MachIndex.h>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <stdexcept>
#include <tuple>
#include <vector>

namespace thirdai::automl::udt::utils::mach {
namespace feat = thirdai::data;

using InputMetrics = bolt::metrics::InputMetrics;
using ComputationPtr = bolt::ComputationPtr;
using CallbackPtr = bolt::callbacks::CallbackPtr;
using DistributedCommPtr = bolt::DistributedCommPtr;

static auto allColumns(const std::string& input_indices_column,
                       const std::string& input_values_column,
                       const std::string& label_column,
                       const std::string& bucket_column) {
  std::vector<std::string> all_columns{label_column, bucket_column};
  for (const auto& columns : model_input_columns) {
    all_columns.push_back(columns.indices());
    if (columns.values()) {
      all_columns.push_back(columns.values().value());
    }
  }
  return all_columns;
}

/**
 * Mach is a classifier that is agnostic to how inputs are featurized.
 * Its methods expect column maps or column map iterators as input so that it
 * can apply additional transformations when necessary. Column maps are always
 * expected to have "model input columns" [1] and sometimes expected to have a
 * "label column" [2] or a "bucket column" [3].
 *
 * [1] "model input columns" are columns that contain indices and values
 * that will be passed into BOLT as input.
 * [2] A "label column" contains integer IDs / labels since Mach currently only
 * supports integer labels.
 * [3] A "bucket column" contains integers representing Mach buckets.
 */
class Mach {
 public:
  Mach(const bolt::Model& model, uint32_t num_hashes,
       float mach_sampling_threshold, bool freeze_hash_tables,
       data::OutputColumnsList model_input_columns, std::string label_column,
       std::string bucket_column, bool use_rlhf, uint32_t num_balancing_docs,
       uint32_t num_balancing_samples_per_doc);

  static auto make(const bolt::Model& model, uint32_t num_hashes,
                   float mach_sampling_threshold, bool freeze_hash_tables,
                   data::OutputColumnsList model_input_columns,
                   std::string label_column, std::string bucket_column,
                   bool use_rlhf, uint32_t num_balancing_docs,
                   uint32_t num_balancing_samples_per_doc) {
    return std::make_shared<Mach>(
        model, num_hashes, mach_sampling_threshold, freeze_hash_tables,
        std::move(model_input_columns), std::move(label_column),
        std::move(bucket_column), use_rlhf, num_balancing_docs,
        num_balancing_samples_per_doc);
  }

  void randomlyAssignBuckets(uint32_t num_entities) {
    index()->randomlyAssignBuckets(num_entities);
  }

  /**
   * `table` is expected to have bolt input columns and a doc id column.
   */
  void introduceEntities(const feat::ColumnMap& columns,
                         std::optional<uint32_t> num_buckets_to_sample_opt,
                         uint32_t num_random_hashes);

  void eraseEntity(uint32_t entity) {
    index()->erase(entity);
    _state->labelwiseSamples()->removeEntity(entity);

    if (index()->numEntities() == 0) {
      std::cout << "Warning. Every learned class has been forgotten. The model "
                   "will currently return nothing on calls to evaluate, "
                   "predict, or predictBatch."
                << std::endl;
    }

    updateSamplingStrategy();
  }

  void eraseAllEntities() {
    index()->clear();
    _state->labelwiseSamples()->clear();
    updateSamplingStrategy();
  }

  /**
   * Tables returned by `train_iter` and `valid_iter` are expected to have bolt
   * input columns and a doc id column.
   */
  bolt::metrics::History train(feat::ColumnMapIteratorPtr train_iter,
                               feat::ColumnMapIteratorPtr val_iter,
                               float learning_rate, uint32_t epochs,
                               const InputMetrics& train_metrics,
                               const InputMetrics& val_metrics,
                               const std::vector<CallbackPtr>& callbacks,
                               TrainOptions options,
                               const DistributedCommPtr& comm);

  /**
   * `table` is expected to have bolt input columns and a doc id column.
   */
  void train(feat::ColumnMap columns, float learning_rate);

  void trainBuckets(feat::ColumnMap columns, float learning_rate);

  /**
   * Tables returned by `eval_iter` are expected to have bolt input columns and
   * a doc id column.
   */
  bolt::metrics::History evaluate(feat::ColumnMapIteratorPtr eval_iter,
                                  const bolt::metrics::InputMetrics& metrics,
                                  bool sparse_inference, bool verbose);

  /**
   * `table` is expected to have bolt input columns.
   */
  std::vector<std::vector<std::pair<uint32_t, double>>> predict(
      const feat::ColumnMap& columns, bool sparse_inference, uint32_t top_k,
      uint32_t num_scanned_buckets);

  std::vector<std::vector<uint32_t>> predictBuckets(
      const feat::ColumnMap& columns, bool sparse_inference,
      std::optional<uint32_t> top_k, bool force_non_empty);

  bolt::metrics::History associateTrain(
      feat::ColumnMap from_table, const feat::ColumnMap& to_table,
      data::ColumnMap train_data, float learning_rate, uint32_t repeats,
      uint32_t num_buckets, uint32_t epochs, size_t batch_size,
      const InputMetrics& metrics, TrainOptions options);

  /**
   * `from_table` and `to_table` are expected to have bolt input columns.
   * `balancing_table` is expected to have bolt input columns and a doc id
   * column.
   */
  void associate(feat::ColumnMap from_table, const feat::ColumnMap& to_table,
                 float learning_rate, uint32_t repeats, uint32_t num_balancers,
                 uint32_t num_buckets, uint32_t epochs, size_t batch_size);

  /**
   * `upvotes` and `balancers` are expected to have bolt input columns and a doc
   * id column.
   */
  void upvote(feat::ColumnMap upvotes, float learning_rate, uint32_t repeats,
              uint32_t num_balancers, uint32_t epochs, size_t batch_size);

  /**
   * `table` is expected to have bolt input columns.
   */
  std::vector<uint32_t> outputCorrectness(const feat::ColumnMap& columns,
                                          const std::vector<uint32_t>& labels,
                                          std::optional<uint32_t> num_hashes,
                                          bool sparse_inference);

  /**
   * `table` is expected to have bolt input columns.
   */
  bolt::TensorPtr embedding(const feat::ColumnMap& columns);

  std::vector<float> entityEmbedding(uint32_t entity) const;

  ModelPtr& model() { return _model; }

  dataset::mach::MachIndexPtr& index() { return _state->machIndex(); }

  const dataset::mach::MachIndexPtr& index() const {
    return _state->machIndex();
  }

  void setIndex(dataset::mach::MachIndexPtr new_index) {
    _state->setMachIndex(std::move(new_index));
    updateSamplingStrategy();
  }

  void setMachSamplingThreshold(float threshold) {
    _mach_sampling_threshold = threshold;
    updateSamplingStrategy();
  }

  void enableRlhf(uint32_t num_balancing_docs,
                  uint32_t num_balancing_samples_per_doc) {
    if (_state->labelwiseSamples()) {
      std::cout << "rlhf already enabled." << std::endl;
      return;
    }
    std::vector<std::string> all_columns;
    for (const auto& columns : _all_bolt_columns) {
      all_columns.push_back(columns.indices());
      if (columns.values()) {
        all_columns.push_back(columns.values().value());
      }
    }
    _state->enableLabelwiseSamples(num_balancing_docs,
                                   num_balancing_samples_per_doc, labelColumn(),
                                   all_columns);
    _rlhf_sampler = data::RLHFSampler::make();
  }

  bool rlhfEnabled() { return !!_state->labelwiseSamples(); }

 private:
  void updateSamplingStrategy();

  void teach(feat::ColumnMap feedback, float learning_rate,
             uint32_t feedback_repetitions, uint32_t num_balancers,
             uint32_t epochs, size_t batch_size);

  data::LoaderPtr getDataLoader(data::ColumnMapIteratorPtr data,
                                bool store_balancers, size_t batch_size,
                                bool shuffle, bool verbose,
                                dataset::DatasetShuffleConfig shuffle_config =
                                    dataset::DatasetShuffleConfig()) {
    auto transformations = data::Pipeline::make({_label_to_buckets});
    if (store_balancers && _rlhf_sampler) {
      transformations = transformations->then(_rlhf_sampler);
    }
    return data::Loader::make(
        std::move(data), transformations, _state, _bolt_input_columns,
        _bolt_label_columns, batch_size, shuffle, verbose,
        shuffle_config.min_buffer_size, shuffle_config.seed);
  }

  data::ColumnMap associateSamples(data::ColumnMap from_columns,
                                   const data::ColumnMap& to_columns,
                                   uint32_t num_buckets) {
    auto mach_labels = thirdai::data::ArrayColumn<uint32_t>::make(
        predictBuckets(to_columns, /* sparse_inference= */ false, num_buckets,
                       /* force_non_empty= */ true),
        index()->numBuckets());
    from_columns.setColumn(bucketColumn(), mach_labels);
    addDummyLabels(from_columns);
    return from_columns;
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

  void addRlhfSamplesIfNeeded(const data::ColumnMap& columns) const {
    if (_rlhf_sampler) {
      _rlhf_sampler->applyStateless(columns);
    }
  }

  data::ColumnMap keepBoltColumns(data::ColumnMap&& columns) const {
    return data::keepColumns(std::move(columns), _all_bolt_columns);
  }

  bolt::TensorList inputTensors(const data::ColumnMap& columns) const {
    return data::toTensors(columns, _bolt_input_columns);
  }

  bolt::TensorList labelTensors(const data::ColumnMap& columns) const {
    return data::toTensors(columns, _bolt_label_columns);
  }

  const std::string& bucketColumn() const {
    return _bolt_label_columns.front().indices();
  }

  const std::string& labelColumn() const {
    return _bolt_label_columns.back().indices();
  }

  Mach() {}  // for cereal

  bolt::ModelPtr _model;
  bolt::ComputationPtr _emb;
  float _mach_sampling_threshold;
  bool _freeze_hash_tables;

  feat::StatePtr _state;

  data::TransformationPtr _label_to_buckets;
  data::TransformationPtr _rlhf_sampler;
  data::OutputColumnsList _bolt_input_columns;
  data::OutputColumnsList _bolt_label_columns;
  data::OutputColumnsList _all_bolt_columns;

  friend class cereal::access;
  template <typename Archive>
  void serialize(Archive& archive);
};

using MachPtr = std::shared_ptr<Mach>;

}  // namespace thirdai::automl::udt::utils::mach
