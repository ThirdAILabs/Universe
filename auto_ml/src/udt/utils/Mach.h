#pragma once

#include "MachDetails.h"
#include <bolt/python_bindings/CtrlCCheck.h>
#include <bolt/src/neuron_index/MachNeuronIndex.h>
#include <bolt/src/nn/model/Model.h>
#include <bolt/src/nn/ops/FullyConnected.h>
#include <bolt/src/nn/ops/Op.h>
#include <bolt/src/nn/tensor/Tensor.h>
#include <bolt/src/train/metrics/Metric.h>
#include <auto_ml/src/config/ModelConfig.h>
#include <auto_ml/src/featurization/LiteFeat.h>
#include <auto_ml/src/featurization/ReservedColumns.h>
#include <auto_ml/src/rlhf/RLHFSampler.h>
#include <auto_ml/src/udt/utils/Classifier.h>
#include <auto_ml/src/udt/utils/Models.h>
#include <data/src/ColumnMap.h>
#include <data/src/ColumnMapIterator.h>
#include <data/src/Loader.h>
#include <data/src/TensorConversion.h>
#include <data/src/columns/ArrayColumns.h>
#include <data/src/columns/ValueColumns.h>
#include <data/src/transformations/AddBalancingSamples.h>
#include <data/src/transformations/MachLabel.h>
#include <data/src/transformations/State.h>
#include <data/src/transformations/Transformation.h>
#include <data/src/transformations/TransformationList.h>
#include <dataset/src/mach/MachIndex.h>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <stdexcept>
#include <tuple>

namespace thirdai::automl::udt::utils {

namespace mach = dataset::mach;
namespace feat = thirdai::data;

static bolt::ComputationPtr getEmbeddingComputation(const bolt::Model& model) {
  // This defines the embedding as the second to last computatation in the
  // computation graph.
  auto computations = model.computationOrder();
  return computations.at(computations.size() - 2);
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
  Mach(uint32_t input_dim, uint32_t num_buckets, uint32_t num_hashes,
       float mach_sampling_threshold, bool freeze_hash_tables,
       const config::ArgumentMap& args,
       const std::optional<std::string>& model_config)

      : _model(utils::buildModel(
            /* input_dim= */ input_dim,
            /* output_dim= */ num_buckets,
            /* args= */ args,
            /* model_config= */ model_config,
            /* use_sigmoid_bce= */ true,
            /* mach= */ true)),
        _emb(getEmbeddingComputation(*_model)),
        _state(feat::State::make(mach::MachIndex::make(
            /* num_buckets= */ num_buckets,
            /* num_hashes=*/num_hashes))),
        _mach_sampling_threshold(mach_sampling_threshold),
        _freeze_hash_tables(freeze_hash_tables)

  {
    updateSamplingStrategy();
  }

  static auto make(uint32_t input_dim, uint32_t num_buckets,
                   uint32_t num_hashes, float mach_sampling_threshold,
                   bool freeze_hash_tables, const config::ArgumentMap& args,
                   const std::optional<std::string>& model_config) {
    return std::make_shared<Mach>(input_dim, num_buckets, num_hashes,
                                  mach_sampling_threshold, freeze_hash_tables,
                                  args, model_config);
  }

  void randomlyAssignBuckets(uint32_t num_entities) {
    index()->randomlyAssignBuckets(num_entities);
  }

  /**
   * `table` is expected to have bolt input columns and a doc id column.
   */
  void introduceEntities(const feat::ColumnMap& table,
                         const feat::OutputColumnsList& input_columns,
                         const std::string& doc_id_column,
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
                               feat::ColumnMapIteratorPtr valid_iter,
                               const feat::OutputColumnsList& input_columns,
                               const std::string& doc_id_column,
                               float learning_rate, uint32_t epochs,
                               const InputMetrics& train_metrics,
                               const InputMetrics& val_metrics,
                               const std::vector<CallbackPtr>& callbacks,
                               TrainOptions options,
                               const bolt::DistributedCommPtr& comm);

  /**
   * `table` is expected to have bolt input columns and a doc id column.
   */
  void trainBatch(feat::ColumnMap table,
                  const feat::OutputColumnsList& input_columns,
                  const std::string& doc_id_column, float learning_rate);

  /**
   * Tables returned by `eval_iter` are expected to have bolt input columns and
   * a doc id column.
   */
  bolt::metrics::History evaluate(feat::ColumnMapIteratorPtr eval_iter,
                                  const feat::OutputColumnsList& input_columns,
                                  const std::string& doc_id_column,
                                  const InputMetrics& metrics,
                                  bool sparse_inference, bool verbose);

  /**
   * `table` is expected to have bolt input columns.
   */
  std::vector<std::vector<std::pair<uint32_t, double>>> predict(
      const feat::ColumnMap& table,
      const feat::OutputColumnsList& input_columns, bool sparse_inference,
      uint32_t top_k, uint32_t num_scanned_buckets);

  /**
   * `from_table` and `to_table` are expected to have bolt input columns.
   * `balancing_table` is expected to have bolt input columns and a doc id
   * column.
   */
  bolt::metrics::History associate(
      feat::ColumnMap from_table, const feat::ColumnMap& to_table,
      feat::ColumnMap balancing_table,
      const feat::OutputColumnsList& input_columns,
      const std::string& doc_id_column, float learning_rate, uint32_t repeats,
      uint32_t num_buckets, uint32_t epochs, size_t batch_size,
      const InputMetrics& metrics = {}, bool verbose = false,
      std::optional<uint32_t> logging_interval = std::nullopt);

  /**
   * `upvotes` and `balancers` are expected to have bolt input columns and a doc
   * id column.
   */
  void upvote(feat::ColumnMap upvotes, feat::ColumnMap balancers,
              const feat::OutputColumnsList& input_columns,
              const std::string& doc_id_column, float learning_rate,
              uint32_t repeats, uint32_t epochs, size_t batch_size);

  /**
   * `table` is expected to have bolt input columns.
   */
  std::vector<uint32_t> outputCorrectness(
      const feat::ColumnMap& table,
      const feat::OutputColumnsList& input_columns,
      const std::vector<uint32_t>& labels, std::optional<uint32_t> num_hashes,
      bool sparse_inference);

  /**
   * `table` is expected to have bolt input columns.
   */
  bolt::TensorPtr embedding(const feat::ColumnMap& table,
                            const feat::OutputColumnsList& input_columns);

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
  static std::tuple<feat::MachLabelPtr, feat::OutputColumnsList> machTransform(
      const std::string& doc_id_column);

  void updateSamplingStrategy();

  utils::ModelPtr _model;
  bolt::ComputationPtr _emb;
  feat::StatePtr _state;
  float _mach_sampling_threshold;
  bool _freeze_hash_tables;
};

using MachPtr = std::shared_ptr<Mach>;

}  // namespace thirdai::automl::udt::utils
