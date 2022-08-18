#pragma once

#include "Graph.h"
#include <bolt/src/graph/DatasetContext.h>
#include <bolt/src/graph/ExecutionConfig.h>
#include <bolt/src/graph/InferenceOutputTracker.h>
#include <bolt/src/graph/nodes/FullyConnected.h>
#include <bolt/src/graph/nodes/TokenInput.h>
#include <bolt/src/loss_functions/LossFunctions.h>
#include <bolt/src/metrics/MetricAggregator.h>
#include <dataset/src/Datasets.h>
#include <utility>
#include <vector>

namespace thirdai::bolt {

class DistributedTrainingContext {
 public:
  DistributedTrainingContext(
      std::vector<InputPtr> inputs, NodePtr output,
      const std::vector<dataset::BoltDatasetPtr>& train_data,
      const dataset::BoltDatasetPtr& train_labels,
      const TrainConfig& train_config, std::shared_ptr<LossFunction> loss,
      bool print_when_done)
      : _bolt_graph(std::make_shared<BoltGraph>(
            std::vector<InputPtr>{std::move(inputs)}, output)),
        _train_context(DatasetContext(train_data, {}, train_labels)),
        _learning_rate(train_config.learningRate()),
        _metrics(train_config.getMetricAggregator()),
        _rebuild_hash_tables_batch(
            train_config.getRebuildHashTablesBatchInterval(
                _train_context.batchSize(), _train_context.len())),
        _reconstruct_hash_functions_batch(
            train_config.getReconstructHashFunctionsBatchInterval(
                _train_context.batchSize(), _train_context.len())) {
    _bolt_graph->compile(std::move(loss), print_when_done);
    _bolt_graph->verifyCanTrain(_train_context);
    _bolt_graph->prepareToProcessBatches(_train_context.batchSize(),
                                         /* use_sparsity=*/true);
    _bolt_graph->enableDistributedTraining();
  }

  DistributedTrainingContext(BoltGraphPtr bolt_graph_ptr,
                             const dataset::BoltDatasetPtr& train_data,
                             const dataset::BoltDatasetPtr& train_labels,
                             const TrainConfig& train_config)
      : _bolt_graph(std::move(bolt_graph_ptr)),
        _train_context(DatasetContext({train_data}, {}, train_labels)),
        _learning_rate(train_config.learningRate()),
        _metrics(train_config.getMetricAggregator()),
        _rebuild_hash_tables_batch(
            train_config.getRebuildHashTablesBatchInterval(
                _train_context.batchSize(), _train_context.len())),
        _reconstruct_hash_functions_batch(
            train_config.getReconstructHashFunctionsBatchInterval(
                _train_context.batchSize(), _train_context.len())) {
    _bolt_graph->verifyCanTrain(_train_context);
    _bolt_graph->prepareToProcessBatches(_train_context.batchSize(),
                                         /* use_sparsity=*/true);
    _bolt_graph->enableDistributedTraining();
  }

  void calculateGradientSingleNode(uint32_t batch_idx);

  void updateParametersSingleNode() const;

  uint64_t numTrainingBatches() const;

  void finishTraining() const;

  NodePtr getNodeByName(const std::string& node_name) const;

  BoltGraphPtr _bolt_graph;

 private:
  DatasetContext _train_context;
  float _learning_rate;
  MetricAggregator _metrics;

  uint32_t _rebuild_hash_tables_batch;
  uint32_t _reconstruct_hash_functions_batch;
};
using DistributedTrainingContextptr =
    std::shared_ptr<DistributedTrainingContext>;

}  // namespace thirdai::bolt