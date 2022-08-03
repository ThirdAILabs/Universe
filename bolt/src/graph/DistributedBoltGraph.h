#pragma once

#include "Graph.h"
#include <bolt/src/graph/DatasetContext.h>
#include <bolt/src/graph/ExecutionConfig.h>
#include <bolt/src/graph/InferenceOutputTracker.h>
#include <bolt/src/graph/nodes/TokenInput.h>
#include <bolt/src/loss_functions/LossFunctions.h>
#include <bolt/src/metrics/MetricAggregator.h>
#include <utility>
#include <vector>

namespace thirdai::bolt {

class DistributedGraph {
 public:
  DistributedGraph(std::vector<InputPtr> inputs, NodePtr output,
                   const std::vector<dataset::BoltDatasetPtr>& train_data,
                   const dataset::BoltDatasetPtr& train_labels,
                   const TrainConfig& train_config,
                   std::shared_ptr<LossFunction> loss, bool print_when_done)
      : DistributedBoltGraph(BoltGraph(std::move(inputs), std::move(output))),
        train_context(DatasetContext(train_data, {}, train_labels)),
        learning_rate(train_config.learningRate()),
        metrics(train_config.getMetricAggregator()),
        rebuild_hash_tables_batch(
            train_config.getRebuildHashTablesBatchInterval(
                train_context.batchSize(), train_context.len())),
        reconstruct_hash_functions_batch(
            train_config.getReconstructHashFunctionsBatchInterval(
                train_context.batchSize(), train_context.len())) {
    DistributedBoltGraph.compile(std::move(loss), print_when_done);
    DistributedBoltGraph.verifyCanTrain(train_context);
    DistributedBoltGraph.prepareToProcessBatches(train_context.batchSize(),
                                                 /* use_sparsity=*/true);
    DistributedBoltGraph.isDistributedTraining();
  }

  void calculateGradientSingleNode(uint32_t batch_idx);

  void updateParametersSingleNode();
  uint64_t num_of_training_batch() const;

  InferenceResult predict(
      const std::vector<dataset::BoltDatasetPtr>& test_data,
      const std::vector<dataset::BoltTokenDatasetPtr>& test_tokens,
      const dataset::BoltDatasetPtr& test_labels,
      const PredictConfig& predict_config);

  NodePtr getNodeByName(const std::string& node_name) const;

  BoltGraph DistributedBoltGraph;
  DatasetContext train_context;
  float learning_rate;
  MetricAggregator metrics;

  uint32_t rebuild_hash_tables_batch;
  uint32_t reconstruct_hash_functions_batch;
};

}  // namespace thirdai::bolt