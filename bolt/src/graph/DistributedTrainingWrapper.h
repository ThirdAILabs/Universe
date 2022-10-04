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

class DistributedTrainingWrapper {
 public:
  explicit DistributedTrainingWrapper(BoltGraphPtr model)
      : _model(std::move(model)) {}

  void freezeHashTables() {
    _model->freezeHashTables(/* insert_labels_if_not_found = */ true);
  }

  void cleanupAfterBatchProcessing() { _model->cleanupAfterBatchProcessing(); }

  BoltGraphPtr model() { return _model; }

 protected:
  BoltGraphPtr _model;

 private:
  virtual bool computeAndSaveNextBatchGradients() = 0;

  virtual void updateParameters() = 0;

  virtual void moveToNextEpoch() = 0;
};

class DistributedTabularTrainingWrapper;
using DistributedTabularTrainingWrapperPtr =
    std::shared_ptr<DistributedTabularTrainingWrapper>;

class DistributedInMemoryTrainingWrapper;
using DistributedInMemoryTrainingWrapperPtr =
    std::shared_ptr<DistributedInMemoryTrainingWrapper>;

class DistributedTabularTrainingWrapper final
    : public DistributedTrainingWrapper {
 public:
  explicit DistributedTabularTrainingWrapper(
      BoltGraphPtr model, TrainConfig train_config,
      automl::deployment_config::DatasetLoaderFactoryPtr factory,
      dataset::DataLoaderPtr loader, uint32_t max_in_memory_batches)
      : DistributedTrainingWrapper(std::move(model)),
        _dataset_loader_factory(std::move(factory)),
        _data_loader(std::move(loader)),
        _still_on_initial_dataset(false),
        _out_of_batches(true),
        _batch_idx_within_train_context(0),
        _max_in_memory_batches(max_in_memory_batches),
        _train_config(std::move(train_config)),
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

  // Returns false if at the end of this function we are out of batches
  bool computeAndSaveNextBatchGradients() final {
    if (_out_of_batches) {
      throw std::runtime_error(
          "Cannot compute next batch of gradients because we are out of "
          "batches");
    }

    _current_train_context->setInputs(_batch_idx_within_train_context,
                                      _model->_inputs);
    const BoltBatch& batch_labels =
        _current_train_context->labels()->at(_batch_idx_within_train_context);
    _model->processTrainingBatch(batch_labels, _metric_aggregator);

    moveToNextBatch();

    return !_out_of_batches;
  }

  void updateParameters() final {
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

  void moveToNextEpoch() final {
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

 private:
  automl::deployment_config::DatasetLoaderFactoryPtr _dataset_loader_factory;
  dataset::DataLoaderPtr _data_loader;
  bool _still_on_initial_dataset;
  bool _out_of_batches;
  uint64_t _batch_idx_within_train_context;
  // This should always have a non null value after the first load
  DatasetContextPtr _current_train_context;
  uint32_t _max_in_memory_batches;

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

class DistributedInMemoryTrainingWrapper final
    : public DistributedTrainingWrapper {
 public:
  explicit DistributedInMemoryTrainingWrapper(
      BoltGraphPtr model, const dataset::BoltDatasetList& train_data,
      const dataset::BoltDatasetPtr& train_labels, TrainConfig train_config)
      : DistributedTrainingWrapper(std::move(model)),
        _train_context(DatasetContext(train_data, train_labels)),
        _train_config(std::move(train_config)),
        _metric_aggregator(_train_config.getMetricAggregator()),
        _current_batch_index(0) {
    _model->verifyCanTrain(_train_context);
    _model->prepareToProcessBatches(_train_context.batchSize(),
                                    /* use_sparsity=*/true);
    _model->enableDistributedTraining();
  }

  bool computeAndSaveNextBatchGradients() final {
    if (_current_batch_index >= _train_context.numBatches()) {
      throw std::runtime_error(
          "Cannot compute next batch of gradients because we are out of "
          "batches");
    }

    _train_context.setInputs(_current_batch_index, _model->_inputs);
    const BoltBatch& batch_labels =
        _train_context.labels()->at(_current_batch_index);
    _model->processTrainingBatch(batch_labels, _metric_aggregator);

    _current_batch_index++;

    return _current_batch_index < _train_context.numBatches();
  }

  void moveToNextEpoch() final { _current_batch_index = 0; }

  void updateParameters() final {
    _model->updateParametersAndSampling(
        /* learning_rate = */ _train_config.learningRate(),
        /* rebuild_hash_tables_batch = */
        _train_config.getRebuildHashTablesBatchInterval(
            _train_context.batchSize(), _train_context.len()),
        /* reconstruct_hash_functions_batch = */
        _train_config.getReconstructHashFunctionsBatchInterval(
            _train_context.batchSize(), _train_context.len()));
  }

 private:
  DatasetContext _train_context;
  TrainConfig _train_config;
  MetricAggregator _metric_aggregator;
  uint64_t _current_batch_index;
};

}  // namespace thirdai::bolt