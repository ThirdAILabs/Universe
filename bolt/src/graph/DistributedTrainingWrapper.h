#pragma once

#include "Graph.h"
#include <bolt/src/callbacks/Callback.h>
#include <bolt/src/graph/DatasetContext.h>
#include <bolt/src/graph/ExecutionConfig.h>
#include <bolt/src/graph/InferenceOutputTracker.h>
#include <bolt/src/graph/nodes/FullyConnected.h>
#include <bolt/src/loss_functions/LossFunctions.h>
#include <bolt/src/metrics/MetricAggregator.h>
#include <utility>
#include <vector>

namespace thirdai::bolt {

class DistributedTrainingWrapper {
 public:
  DistributedTrainingWrapper(BoltGraphPtr bolt_graph,
                             const dataset::BoltDatasetList& train_data,
                             const dataset::BoltDatasetPtr& train_labels,
                             TrainConfig train_config)
      : _bolt_graph(std::move(bolt_graph)),
        _train_context(DatasetContext(train_data, train_labels)),
        _train_config(std::move(train_config)),
        _metric_aggregator(_train_config.getMetricAggregator()) {
    _bolt_graph->verifyCanTrain(_train_context);
    _bolt_graph->prepareToProcessBatches(_train_context.batchSize(),
                                         /* use_sparsity=*/true);
    _bolt_graph->disableSparseParameterUpdates();
    const auto& validation = _train_config.getValidationContext();
    if (validation) {
      _bolt_graph->_tracked_metric = validation->metric();
      if (_bolt_graph->_tracked_metric != nullptr) {
        _bolt_graph->_best_validation_metric =
            _bolt_graph->_tracked_metric->worst();
      }
    }
  }

  void computeAndSaveBatchGradients(uint32_t batch_idx) {
    _train_context.setInputs(batch_idx, _bolt_graph->_inputs);
    const BoltBatch& batch_labels = _train_context.labels()->at(batch_idx);
    _bolt_graph->processTrainingBatch(batch_labels, _metric_aggregator);
  }

  void updateParameters() {
    _bolt_graph->updateParametersAndSampling(
        /* learning_rate = */ _train_config.learningRate(),
        /* rebuild_hash_tables_batch = */
        _train_config.getRebuildHashTablesBatchInterval(
            _train_context.batchSize(), _train_context.len()),
        /* reconstruct_hash_functions_batch = */
        _train_config.getReconstructHashFunctionsBatchInterval(
            _train_context.batchSize(), _train_context.len()));
    _bolt_graph->logValidateAndSave(_train_context.batchSize(), _train_config,
                                    _metric_aggregator);
  }

  BoltGraphPtr getModel() { return _bolt_graph; }

  void finishTraining() { _bolt_graph->cleanupAfterBatchProcessing(); }

 private:
  BoltGraphPtr _bolt_graph;
  DatasetContext _train_context;
  TrainConfig _train_config;
  MetricAggregator _metric_aggregator;
};

}  // namespace thirdai::bolt