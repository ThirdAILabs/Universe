#pragma once

#include "MachDetails.h"
#include <bolt/python_bindings/CtrlCCheck.h>
#include <bolt/src/neuron_index/MachNeuronIndex.h>
#include <bolt/src/nn/loss/BinaryCrossEntropy.h>
#include <bolt/src/nn/loss/CategoricalCrossEntropy.h>
#include <bolt/src/nn/model/Model.h>
#include <bolt/src/nn/ops/FullyConnected.h>
#include <bolt/src/nn/ops/Input.h>
#include <bolt/src/nn/ops/Op.h>
#include <bolt/src/nn/tensor/Tensor.h>
#include <bolt/src/train/callbacks/Callback.h>
#include <bolt/src/train/metrics/Metric.h>
#include <bolt/src/train/trainer/DistributedComm.h>
#include <auto_ml/src/featurization/ReservedColumns.h>
#include <auto_ml/src/featurization/UDTTransformationFactory.h>
#include <auto_ml/src/rlhf/RLHFSampler.h>
#include <auto_ml/src/udt/UDTBackend.h>
#include <data/src/ColumnMap.h>
#include <data/src/ColumnMapIterator.h>
#include <data/src/Loader.h>
#include <data/src/TensorConversion.h>
#include <data/src/columns/ArrayColumns.h>
#include <data/src/columns/ValueColumns.h>
#include <data/src/transformations/AddBalancingSamples.h>
#include <data/src/transformations/MachLabel.h>
#include <data/src/transformations/Pipeline.h>
#include <data/src/transformations/State.h>
#include <data/src/transformations/Transformation.h>
#include <dataset/src/dataset_loaders/DatasetLoader.h>
#include <dataset/src/mach/MachIndex.h>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <stdexcept>
#include <tuple>
#include <vector>

namespace thirdai::automl::udt::utils {

namespace mach = dataset::mach;
namespace feat = thirdai::data;

using InputMetrics = bolt::metrics::InputMetrics;
using ComputationPtr = bolt::ComputationPtr;
using CallbackPtr = bolt::callbacks::CallbackPtr;
using DistributedCommPtr = bolt::DistributedCommPtr;

static bolt::ComputationPtr getEmbeddingComputation(const bolt::Model& model) {
  // This defines the embedding as the second to last computatation in the
  // computation graph.
  auto computations = model.computationOrder();
  return computations.at(computations.size() - 2);
}

static auto machModel(const bolt::Model& model) {
  if (model.outputs().size() > 1 || model.labels().size() > 1 ||
      model.losses().size() > 1) {
    throw std::runtime_error(
        "Mach currently only supports models with a single output, a single "
        "label, and a single loss function.");
  }
  return bolt::Model::make(
      model.inputs(), model.outputs(), model.losses(),
      /* additional_labels= */
      {bolt::Input::make(std::numeric_limits<uint32_t>::max())});
}

static auto allColumns(const data::OutputColumnsList& model_input_columns,
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

static auto valueFill(const bolt::Model& model) {
  return model.losses().front()->logitsSumToOne()
             ? data::ValueFillType::SumToOne
             : data::ValueFillType::Ones;
}

/**
 * Mach is a classifier that is agnostic to how inputs are featurized.
 * Its methods expect column maps (often called tables for brevity) or column
 * map iterators as input so that it can apply additional transformations when
 * necessary. Column maps are always expected to have "bolt input columns" [1]
 * and sometimes expected to have a "doc id column" [2].
 *
 * [1] "bolt input columns" are columns that contain indices and values
 * that will be passed into BOLT as input.
 * [2] A "doc id column" contains integer IDs / labels since Mach currently only
 * supports integer labels.
 */
class Mach {
 public:
  Mach(const bolt::Model& model, uint32_t num_hashes,
       float mach_sampling_threshold, bool freeze_hash_tables,
       data::OutputColumnsList model_input_columns, std::string label_column,
       std::string bucket_column)

      : _model(machModel(model)),
        _emb(getEmbeddingComputation(*_model)),
        _mach_sampling_threshold(mach_sampling_threshold),
        _freeze_hash_tables(freeze_hash_tables),
        _state(feat::State::make(mach::MachIndex::make(
            /* num_buckets= */ model.outputs().front()->dim(),
            /* num_hashes=*/num_hashes))),
        _label_to_buckets(data::MachLabel::make(label_column, bucket_column)),
        _store_rlhf_samples(data::AddBalancingSamples::make(
            allColumns(model_input_columns, label_column, bucket_column))),
        _bolt_input_columns(std::move(model_input_columns)),
        _bolt_label_columns(
            {data::OutputColumns(std::move(bucket_column), valueFill(model)),
             data::OutputColumns(std::move(label_column))}) {
    updateSamplingStrategy();
    for (const auto& columns : _bolt_input_columns) {
      _all_bolt_columns.push_back(columns);
    }
    for (const auto& columns : _bolt_label_columns) {
      _all_bolt_columns.push_back(columns);
    }
  }

  static auto make(const bolt::Model& model, uint32_t num_hashes,
                   float mach_sampling_threshold, bool freeze_hash_tables,
                   data::OutputColumnsList model_input_columns,
                   std::string label_column, std::string bucket_column) {
    return std::make_shared<Mach>(
        model, num_hashes, mach_sampling_threshold, freeze_hash_tables,
        std::move(model_input_columns), std::move(label_column),
        std::move(bucket_column));
  }

  void randomlyAssignBuckets(uint32_t num_entities) {
    index()->randomlyAssignBuckets(num_entities);
  }

  /**
   * `table` is expected to have bolt input columns and a doc id column.
   */
  void introduceEntities(const feat::ColumnMap& table,
                         std::optional<uint32_t> num_buckets_to_sample_opt,
                         uint32_t num_random_hashes);

  void eraseEntity(uint32_t entity) {
    index()->erase(entity);

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
  void train(feat::ColumnMap table, float learning_rate);

  void trainBuckets(feat::ColumnMap table, float learning_rate);

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
      const feat::ColumnMap& table, bool sparse_inference, uint32_t top_k,
      uint32_t num_scanned_buckets);

  std::vector<std::vector<uint32_t>> predictBuckets(
      const feat::ColumnMap& table, bool sparse_inference,
      std::optional<uint32_t> top_k);

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
  std::vector<uint32_t> outputCorrectness(const feat::ColumnMap& table,
                                          const std::vector<uint32_t>& labels,
                                          std::optional<uint32_t> num_hashes,
                                          bool sparse_inference);

  /**
   * `table` is expected to have bolt input columns.
   */
  bolt::TensorPtr embedding(const feat::ColumnMap& table);

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

 private:
  data::LoaderPtr getDataLoader(data::ColumnMapIteratorPtr data,
                                bool store_balancers, size_t batch_size,
                                bool shuffle, bool verbose,
                                dataset::DatasetShuffleConfig shuffle_config =
                                    dataset::DatasetShuffleConfig()) {
    auto transformations = data::Pipeline::make({_label_to_buckets});
    if (store_balancers) {
      transformations = transformations->then(_store_rlhf_samples);
    }
    return data::Loader::make(
        std::move(data), transformations, _state, _bolt_input_columns,
        _bolt_label_columns, batch_size, shuffle, verbose,
        shuffle_config.min_buffer_size, shuffle_config.seed);
  }

  void updateSamplingStrategy();

  void teach(feat::ColumnMap feedback, float learning_rate,
             uint32_t feedback_repetitions, uint32_t num_balancers,
             uint32_t epochs, size_t batch_size);

  const auto& labelColumn() const {
    return _bolt_label_columns.back().indices();
  }

  data::ColumnMap addDummyLabels(data::ColumnMap columns) const {
    auto doc_ids = thirdai::data::ValueColumn<uint32_t>::make(
        std::vector<uint32_t>(columns.numRows(), 0),
        std::numeric_limits<uint32_t>::max());
    columns.setColumn(labelColumn(), doc_ids);
    return columns;
  }

  bolt::ModelPtr _model;
  bolt::ComputationPtr _emb;
  float _mach_sampling_threshold;
  bool _freeze_hash_tables;

  feat::StatePtr _state;

  data::TransformationPtr _label_to_buckets;
  data::TransformationPtr _store_rlhf_samples;
  data::OutputColumnsList _bolt_input_columns;
  data::OutputColumnsList _bolt_label_columns;
  data::OutputColumnsList _all_bolt_columns;
};

using MachPtr = std::shared_ptr<Mach>;

}  // namespace thirdai::automl::udt::utils
