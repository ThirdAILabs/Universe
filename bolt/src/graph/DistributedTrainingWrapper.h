#pragma once

#include "Graph.h"
#include <bolt/src/callbacks/Callback.h>
#include <bolt/src/graph/DatasetContext.h>
#include <bolt/src/graph/ExecutionConfig.h>
#include <bolt/src/graph/InferenceOutputTracker.h>
#include <bolt/src/graph/nodes/FullyConnected.h>
#include <bolt/src/loss_functions/LossFunctions.h>
#include <bolt/src/metrics/MetricAggregator.h>
#include <optional>
#include <stdexcept>
#include <utility>
#include <vector>

namespace thirdai::bolt {

class DistributedTrainingWrapper {
 public:
  DistributedTrainingWrapper(BoltGraphPtr bolt_graph, TrainConfig train_config,
                             uint32_t worker_id)
      : _bolt_graph(std::move(bolt_graph)),
        _train_context(std::nullopt),
        _train_config(std::move(train_config)),
        _metric_aggregator(_train_config.getMetricAggregator()),
        _worker_id(worker_id) {
    _bolt_graph->disableSparseParameterUpdates();
  }

  void computeAndStoreBatchGradients(uint32_t batch_idx) {
    requireTrainContext();
    _train_context->setInputs(batch_idx, _bolt_graph->_inputs);
    const BoltBatch& batch_labels = _train_context->labels()->at(batch_idx);
    _bolt_graph->processTrainingBatch(batch_labels, _metric_aggregator);
  }

  void saveWithOptimizer(bool should_save_optimizer){
    _bolt_graph->saveWithOptimizer(should_save_optimizer);
  }

  void updateParameters() {
    requireTrainContext();
    _bolt_graph->updateParametersAndSampling(
        /* learning_rate = */ _train_config.learningRate(),
        /* rebuild_hash_tables_batch = */
        _train_config.getRebuildHashTablesBatchInterval(
            _train_context->batchSize(), _train_context->len()),
        /* reconstruct_hash_functions_batch = */
        _train_config.getReconstructHashFunctionsBatchInterval(
            _train_context->batchSize(), _train_context->len()));
    if (_worker_id == 0) {
      _bolt_graph->logAndSaveIfNeeded(_train_config, _metric_aggregator);
    }
  }

  std::optional<InferenceMetricData> validationAndSaveBest() {
    return _bolt_graph->validateAndSaveIfBest(
        _train_config, _train_config.getValidationContext().value());
  }

  BoltGraphPtr getModel() { return _bolt_graph; }

  void finishTraining() { requireTrainContext(); }

  uint64_t numBatches() {
    if (!_train_context.has_value()) {
      return 0;
    }
    return _train_context->numBatches();
  }

  void setDatasets(const dataset::BoltDatasetList& train_data,
                   const dataset::BoltDatasetPtr& train_labels) {
    DatasetContext new_context(train_data, train_labels);
    _bolt_graph->verifyCanTrain(new_context);

    if (!_train_context.has_value()) {
      _train_context = new_context;
      return;
    }

    if (new_context.batchSize() != _train_context->batchSize()) {
      throw std::invalid_argument(
          "New datasets must have the same batch size as the original "
          "datasets");
    }

    // Since we have verified that the new DatasetContext has the same batch
    // size as the original, we don't need to call prepareToProcessBatches
    // again.
    _train_context = new_context;
  }

  void freezeHashTables(bool insert_labels_if_not_found) {
    _bolt_graph->freezeHashTables(insert_labels_if_not_found);
  }

  MetricData getUpdatedMetrics() {
    _metric_aggregator.logAndReset();
    auto metric_data = _metric_aggregator.getOutput();
    return metric_data;
  }

 private:
  void requireTrainContext() {
    if (!_train_context.has_value()) {
      throw std::runtime_error(
          "You must call setNewDataasets before you can train the wrapped "
          "model");
    }
  }

  BoltGraphPtr _bolt_graph;
  std::optional<DatasetContext> _train_context;
  TrainConfig _train_config;
  MetricAggregator _metric_aggregator;
  // worker_id here refers to id of this particular bolt graph
  // in distributed context.
  uint32_t _worker_id;
};

}  // namespace thirdai::bolt