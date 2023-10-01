#pragma once

#include "MachDetails.h"
#include <bolt/python_bindings/CtrlCCheck.h>
#include <bolt/src/neuron_index/MachNeuronIndex.h>
#include <bolt/src/nn/model/Model.h>
#include <bolt/src/nn/ops/FullyConnected.h>
#include <bolt/src/nn/ops/Op.h>
#include <bolt/src/nn/tensor/Tensor.h>
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

namespace thirdai::automl::udt::utils {

namespace mach = dataset::mach;
using MachLabel = thirdai::data::MachLabel;
using TransformationList = thirdai::data::TransformationList;
using OutputColumns = thirdai::data::OutputColumns;
using OutputColumnsList = thirdai::data::OutputColumnsList;
using TransformIterator = thirdai::data::TransformIterator;
using TransformedTable = thirdai::data::TransformedTable;
using TransformedIterator = thirdai::data::TransformedIterator;
using TransformedTensors = thirdai::data::TransformedTensors;
using State = thirdai::data::State;
using Loader = thirdai::data::Loader;
using LoaderPtr = thirdai::data::LoaderPtr;

static bolt::ComputationPtr getEmbeddingComputation(const bolt::Model& model) {
  // This defines the embedding as the second to last computatation in the
  // computation graph.
  auto computations = model.computationOrder();
  return computations.at(computations.size() - 2);
}

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
        _mach_index(mach::MachIndex::make(
            /* num_buckets= */ num_buckets,
            /* num_hashes=*/num_hashes)),
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
    _mach_index->randomlyAssignBuckets(num_entities);
  }

  void introduceEntities(const TransformedTable& table,
                         std::optional<uint32_t> num_buckets_to_sample_opt,
                         uint32_t num_random_hashes);

  void eraseEntity(uint32_t entity) {
    _mach_index->erase(entity);

    if (_mach_index->numEntities() == 0) {
      std::cout << "Warning. Every learned class has been forgotten. The model "
                   "will currently return nothing on calls to evaluate, "
                   "predict, or predictBatch."
                << std::endl;
    }

    updateSamplingStrategy();
  }

  void eraseAllEntities() {
    _mach_index->clear();
    updateSamplingStrategy();
  }

  auto train(TransformedIterator train_data,
             std::optional<TransformedIterator> valid_data, float learning_rate,
             uint32_t epochs, const InputMetrics& train_metrics,
             const InputMetrics& val_metrics,
             const std::vector<CallbackPtr>& callbacks, TrainOptions options,
             const bolt::DistributedCommPtr& comm);

  void trainBatch(TransformedTable batch, float learning_rate);

  auto evaluate(TransformedIterator eval_data, const InputMetrics& metrics,
                bool sparse_inference, bool verbose);

  auto predict(const TransformedTable& batch, bool sparse_inference,
               uint32_t top_k, uint32_t num_scanned_buckets);

  auto associate(TransformedTable from_table, const TransformedTable& to_table,
                 TransformedTable balancers, float learning_rate,
                 uint32_t repeats, uint32_t num_buckets, uint32_t epochs,
                 size_t batch_size, const InputMetrics& metrics = {},
                 bool verbose = false,
                 std::optional<uint32_t> logging_interval = std::nullopt);

  void upvote(TransformedTable upvotes, TransformedTable balancers,
              float learning_rate, uint32_t repeats, uint32_t epochs,
              size_t batch_size);

  auto outputCorrectness(const TransformedTable& input,
                         const std::vector<uint32_t>& labels,
                         std::optional<uint32_t> num_hashes,
                         bool sparse_inference);

  auto embedding(const TransformedTable& table);

  auto entityEmbedding(uint32_t entity) const;

  ModelPtr& model() { return _model; }

  auto& index() { return _mach_index; }

  void setIndex(dataset::mach::MachIndexPtr new_index) {
    if (_mach_index && _mach_index->numBuckets() != new_index->numBuckets()) {
      throw std::invalid_argument(
          "Output range mismatch in new index. Index output range should be " +
          std::to_string(_mach_index->numBuckets()) +
          " but provided an index with range = " +
          std::to_string(new_index->numBuckets()) + ".");
    }

    _mach_index = std::move(new_index);

    updateSamplingStrategy();
  }

  void setMachSamplingThreshold(float threshold) {
    _mach_sampling_threshold = threshold;
    updateSamplingStrategy();
  }

 private:
  void updateSamplingStrategy();

  utils::ModelPtr _model;
  bolt::ComputationPtr _emb;
  mach::MachIndexPtr _mach_index;
  float _mach_sampling_threshold;
  bool _freeze_hash_tables;
};

using MachPtr = std::shared_ptr<Mach>;

}  // namespace thirdai::automl::udt::utils