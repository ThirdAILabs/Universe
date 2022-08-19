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

void DistributedTrainingContext::calculateGradientSingleNode(
    uint32_t batch_idx) {
  _train_context.setInputs(batch_idx, _bolt_graph->_inputs,
                           _bolt_graph->_token_inputs);
  const BoltBatch& batch_labels = _train_context.labels()->at(batch_idx);
  _bolt_graph->processTrainingBatch(batch_labels, _metrics);
}

void DistributedTrainingContext::updateParametersSingleNode() const {
  _bolt_graph->updateParametersAndSampling(_learning_rate,
                                           _rebuild_hash_tables_batch,
                                           _reconstruct_hash_functions_batch);
}
uint64_t DistributedTrainingContext::numTrainingBatches() const {
  std::cout << _train_context.numBatches() << std::endl;
  return _train_context.numBatches();
}

void DistributedTrainingContext::finishTraining() const {
  _bolt_graph->cleanupAfterBatchProcessing();
}

NodePtr DistributedTrainingContext::getNodeByName(
    const std::string& node_name) const {
  return _bolt_graph->getNodeByName(node_name);
}

}  // namespace thirdai::bolt