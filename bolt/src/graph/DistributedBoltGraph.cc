#include "DistributedBoltGraph.h"
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

void DistributedGraph::calculateGradientSingleNode(uint32_t batch_idx) {
  train_context.setInputs(batch_idx, DistributedBoltGraph._inputs,
                          DistributedBoltGraph._token_inputs);
  const BoltBatch& batch_labels = train_context.labels()->at(batch_idx);
  DistributedBoltGraph.processTrainingBatch(batch_labels, metrics);
}

void DistributedGraph::updateParametersSingleNode() {
  DistributedBoltGraph.updateParametersandSampling(
      learning_rate, rebuild_hash_tables_batch,
      reconstruct_hash_functions_batch);
}
uint64_t DistributedGraph::num_of_training_batch() const {
  return train_context.numBatches();
}

InferenceResult DistributedGraph::predict(
    const std::vector<dataset::BoltDatasetPtr>& test_data,
    const std::vector<dataset::BoltTokenDatasetPtr>& test_tokens,
    const dataset::BoltDatasetPtr& test_labels,
    const PredictConfig& predict_config) {
  DistributedBoltGraph.cleanupAfterBatchProcessing();
  return DistributedBoltGraph.predict(test_data, test_tokens, test_labels,
                                      predict_config);
}

NodePtr DistributedGraph::getNodeByName(const std::string& node_name) const {
  return DistributedBoltGraph.getNodeByName(node_name);
}

}  // namespace thirdai::bolt