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
                   const TrainConfig& train_config)
      : DistributedBoltGraph(BoltGraph(std::move(inputs), std::move(output))),
        train_context(DatasetContext(train_data, {}, train_labels)),
        train_config(train_config),
        metrics(train_config.getMetricAggregator()),
        rebuild_hash_tables_batch(
            train_config.getRebuildHashTablesBatchInterval(
                train_context.batchSize(), train_context.len())),
        reconstruct_hash_functions_batch(
            train_config.getReconstructHashFunctionsBatchInterval(
                train_context.batchSize(), train_context.len())) {
    DistributedBoltGraph.prepareToProcessBatches(train_context.batchSize(),
                                                 /* use_sparsity=*/true);
    DistributedBoltGraph.isDistributedTraining();
  }

  void compile(std::shared_ptr<LossFunction> loss, bool print_when_done) {
    DistributedBoltGraph.compile(std::move(loss), print_when_done);
  }
  void calculateGradientSingleNode(uint32_t batch_idx) {
    train_context.setInputs(batch_idx, DistributedBoltGraph._inputs,
                            DistributedBoltGraph._token_inputs);

    const BoltBatch& batch_labels = train_context.labels()->at(batch_idx);
    DistributedBoltGraph.processTrainingBatch(batch_labels, metrics);
  }

  void updateParametersSingleNode() {
    DistributedBoltGraph.updateParametersandSampling(
        train_config.learningRate(), rebuild_hash_tables_batch,
        reconstruct_hash_functions_batch);
  }
  uint64_t num_of_training_batch() const { return train_context.numBatches(); }

  InferenceResult predict(const std::vector<dataset::BoltDatasetPtr>& test_data,
                          const dataset::BoltDatasetPtr& test_labels,
                          const PredictConfig& predict_config) {
    return DistributedBoltGraph.predict(test_data, {}, test_labels,
                                        predict_config);
  }

  BoltGraph DistributedBoltGraph;
  DatasetContext train_context;
  const TrainConfig& train_config;
  MetricAggregator metrics;

  uint32_t rebuild_hash_tables_batch;
  uint32_t reconstruct_hash_functions_batch;
};

}  // namespace thirdai::bolt