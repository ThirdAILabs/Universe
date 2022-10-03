#pragma once

#include "Graph.h"
#include <bolt/src/graph/DatasetContext.h>
#include <bolt/src/graph/ExecutionConfig.h>
#include <bolt/src/graph/InferenceOutputTracker.h>
#include <bolt/src/graph/callbacks/Callback.h>
#include <bolt/src/graph/nodes/FullyConnected.h>
#include <bolt/src/loss_functions/LossFunctions.h>
#include <bolt/src/metrics/MetricAggregator.h>
#include <auto_ml/src/ModelPipeline.h>
#include <dataset/src/DataLoader.h>
#include <limits>
#include <stdexcept>
#include <utility>
#include <vector>

namespace thirdai::bolt {

class DistributedTrainingWrapper;
using DistributedTrainingWrapperPtr =
    std::shared_ptr<DistributedTrainingWrapper>;

class DistributedTrainingWrapper {
 public:
  explicit DistributedTrainingWrapper(
      const automl::ModelPipelinePtr& model_pipeline, float learning_rate,
      dataset::DataLoaderPtr loader, uint32_t max_in_memory_batches)
      : _dataset_loader_factory(model_pipeline->_dataset_factory),
        _data_loader(std::move(loader)),
        _still_on_initial_dataset(false),
        _out_of_batches(true),
        _batch_idx_within_train_context(0),
        _max_in_memory_batches(max_in_memory_batches),
        _model(model_pipeline->_model),
        _train_config(model_pipeline->getTrainConfig(
            /* learning_rate = */ learning_rate, /* epochs = */ 1)),
        _metric_aggregator(_train_config.getMetricAggregator()) {
    tryLoadingNextDataset(/* is_first_load = */ true);
    if (_out_of_batches || _current_train_context == NULL ||
        _still_on_initial_dataset == false) {
      throw std::runtime_error(
          "Did not find any batches in the data source, or some other wrapper "
          "failure when reading in first dataset.");
    }

    _model->verifyCanTrain(*_current_train_context);
    _model->prepareToProcessBatches(_current_train_context->batchSize(),
                                    /* use_sparsity=*/true);
    _model->enableDistributedTraining();
  }

  // Returns false if we are out of batches
  bool computeAndSaveNextBatchGradients() {
    if (_out_of_batches) {
      return false;
    }

    _current_train_context->setInputs(_batch_idx_within_train_context,
                                      _model->_inputs);
    const BoltBatch& batch_labels =
        _current_train_context->labels()->at(_batch_idx_within_train_context);
    _model->processTrainingBatch(batch_labels, _metric_aggregator);

    moveToNextBatch();

    return true;
  }

  void freezeHashTables() {
    _model->freezeHashTables(/* insert_labels_if_not_found = */ true);
  };

  void updateParameters() {
    _model->updateParametersAndSampling(
        /* learning_rate = */ _train_config.learningRate(),
        /* rebuild_hash_tables_batch = */
        _train_config.getRebuildHashTablesBatchInterval(
            _current_train_context->batchSize(), _current_train_context->len()),
        /* reconstruct_hash_functions_batch = */
        _train_config.getReconstructHashFunctionsBatchInterval(
            _current_train_context->batchSize(),
            _current_train_context->len()));
  }

  void cleanupAfterBatchProcessing() { _model->cleanupAfterBatchProcessing(); }

  BoltGraphPtr model() { return _model; }

  void moveToNextEpoch() {
    _batch_idx_within_train_context = 0;

    if (!_still_on_initial_dataset) {
      _data_loader->restart();
      tryLoadingNextDataset(/* is_first_load = */ true);
      if (_out_of_batches || _current_train_context == NULL ||
          _still_on_initial_dataset == false) {
        throw std::runtime_error(
            "Did not find any batches in the data source, or some other "
            "wrapper failure when reading in first dataset.");
      }
    }
  }

  static DistributedTrainingWrapperPtr make(
      const automl::ModelPipelinePtr& model_pipeline, float learning_rate,
      dataset::DataLoaderPtr data_loader, uint32_t max_in_memory_batches) {
    return std::make_shared<DistributedTrainingWrapper>(
        /* model_pipeline = */ model_pipeline,
        /* learning_rate = */ learning_rate,
        /* data_loader = */ std::move(data_loader),
        /* max_in_memory_batches = */ max_in_memory_batches);
  }

  static DistributedTrainingWrapperPtr make(
      const automl::ModelPipelinePtr& model_pipeline, float learning_rate,
      const std::string& filename, uint32_t max_in_memory_batches) {
    uint32_t batch_size = model_pipeline->defaultBatchSize();
    auto data_loader =
        std::make_shared<dataset::SimpleFileDataLoader>(filename, batch_size);
    return make(model_pipeline, learning_rate, data_loader,
                max_in_memory_batches);
  }

 private:
  automl::deployment_config::DatasetLoaderFactoryPtr _dataset_loader_factory;
  dataset::DataLoaderPtr _data_loader;
  bool _still_on_initial_dataset;
  bool _out_of_batches;
  uint64_t _batch_idx_within_train_context;
  // This should always have a non null value after the first load
  DatasetContextPtr _current_train_context;
  uint32_t _max_in_memory_batches;

  BoltGraphPtr _model;
  TrainConfig _train_config;
  MetricAggregator _metric_aggregator;

  void moveToNextBatch() {
    if (_batch_idx_within_train_context + 1 <
        _current_train_context->numBatches()) {
      _batch_idx_within_train_context++;
      return;
    }

    tryLoadingNextDataset(/* is_first_load = */ false);
  }

  void tryLoadingNextDataset(bool is_first_load) {
    auto attempted_load =
        _dataset_loader_factory->getLabeledDatasetLoader(_data_loader)
            ->loadInMemory(_max_in_memory_batches);
    if (!attempted_load.has_value()) {
      _out_of_batches = true;
      return;
    }

    auto [data, labels] = attempted_load.value();
    _current_train_context = std::make_shared<DatasetContext>(data, labels);
    _still_on_initial_dataset = is_first_load;
    _out_of_batches = false;
    _batch_idx_within_train_context = 0;
  }
};

}  // namespace thirdai::bolt