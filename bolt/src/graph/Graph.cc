#include "Graph.h"
#include <bolt/src/graph/nodes/Input.h>
#include <bolt/src/layers/BoltVector.h>
#include <bolt/src/metrics/MetricAggregator.h>
#include <bolt/src/utils/ProgressBar.h>
#include <algorithm>
#include <chrono>
#include <optional>
#include <queue>
#include <unordered_set>

namespace thirdai::bolt {

void BoltGraph::compile(std::shared_ptr<LossFunction> loss) {
  _loss = std::move(loss);

  for (auto& node : _nodes) {
    node->initializeParameters();
  }

  for (auto& node : _nodes) {
    auto node_layers = node->getInternalFullyConnectedLayers();
    _sparse_layers.insert(_sparse_layers.end(), node_layers.begin(),
                          node_layers.end());
  }
}

template MetricData BoltGraph::train(
    std::shared_ptr<dataset::InMemoryDataset<BoltBatch>>&,
    const dataset::BoltDatasetPtr&, float, uint32_t, uint32_t, uint32_t,
    const std::vector<std::string>&, bool);

template <typename BATCH_T>
MetricData BoltGraph::train(
    std::shared_ptr<dataset::InMemoryDataset<BATCH_T>>& train_data,
    const dataset::BoltDatasetPtr& train_labels, float learning_rate,
    uint32_t epochs, uint32_t rehash, uint32_t rebuild,
    // Metrics to compute during training
    const std::vector<std::string>& metric_names,
    // Restrict printouts
    bool verbose) {
  uint32_t batch_size = train_data->at(0).getBatchSize();
  uint32_t rebuild_batch = rebuild / batch_size;
  uint32_t rehash_batch = rehash / batch_size;
  uint64_t num_train_batches = train_data->numBatches();

  // Because of how the datasets are read we know that all batches will not have
  // a batch size larger than this so we can just set the batch size here.
  prepareForBatchProcessing(batch_size, /* use_sparsity=*/true);

  std::vector<double> time_per_epoch;

  MetricAggregator metrics(metric_names, verbose);

  for (uint32_t epoch = 0; epoch < epochs; epoch++) {
    if (verbose) {
      std::cout << "\nEpoch " << (_epoch_count + 1) << ':' << std::endl;
    }
    ProgressBar bar(num_train_batches, verbose);
    auto train_start = std::chrono::high_resolution_clock::now();

    for (uint32_t batch = 0; batch < num_train_batches; batch++) {
      BATCH_T& batch_inputs = train_data->at(batch);

      const BoltBatch& batch_labels = train_labels->at(batch);

      processTrainingBatch(batch_inputs, batch_labels, learning_rate, metrics);

      updateSampling(rehash_batch, rebuild_batch);

      bar.increment();
    }

    auto train_end = std::chrono::high_resolution_clock::now();
    int64_t epoch_time = std::chrono::duration_cast<std::chrono::seconds>(
                             train_end - train_start)
                             .count();

    time_per_epoch.push_back(static_cast<double>(epoch_time));
    if (verbose) {
      std::cout << std::endl
                << "Processed " << num_train_batches << " training batches in "
                << epoch_time << " seconds" << std::endl;
    }
    _epoch_count++;
    metrics.logAndReset();
  }

  auto metric_data = metrics.getOutput();
  metric_data["epoch_times"] = std::move(time_per_epoch);

  return metric_data;
}

template <>
void BoltGraph::processTrainingBatch(BoltBatch& batch_inputs,
                                     const BoltBatch& batch_labels,
                                     float learning_rate,
                                     MetricAggregator& metrics) {
  _inputs[0]->setInputs(&batch_inputs);

#pragma omp parallel for default(none) \
    shared(batch_inputs, batch_labels, metrics)
  for (uint32_t vec_id = 0; vec_id < batch_inputs.getBatchSize(); vec_id++) {
    forward(vec_id, &batch_labels[vec_id]);

    _loss->lossGradients(_output->getOutput(vec_id), batch_labels[vec_id],
                         batch_inputs.getBatchSize());

    backpropagate(vec_id);

    metrics.processSample(_output->getOutput(vec_id), batch_labels[vec_id]);
  }

  updateParameters(learning_rate, ++_batch_cnt);
}

void BoltGraph::updateSampling(uint32_t rehash_batch, uint32_t rebuild_batch) {
  if (checkBatchInterval(rebuild_batch)) {
    rebuildHashFunctions();
    rebuildHashTables();
  } else if (checkBatchInterval(rehash_batch)) {
    rebuildHashTables();
  }
}

template InferenceMetricData BoltGraph::predict(
    const std::shared_ptr<dataset::InMemoryDataset<BoltBatch>>&,
    const dataset::BoltDatasetPtr&, const std::vector<std::string>&, bool,
    uint32_t);

template <typename BATCH_T>
InferenceMetricData BoltGraph::predict(
    // Test dataset
    const std::shared_ptr<dataset::InMemoryDataset<BATCH_T>>& test_data,
    // Test labels
    const dataset::BoltDatasetPtr& test_labels,
    // Metrics to compute
    const std::vector<std::string>& metric_names,
    // Restrict printouts
    bool verbose,
    // Limit the number of batches used in the dataset
    uint32_t batch_limit) {
  bool compute_metrics = test_labels != nullptr;

  uint32_t batch_size = test_data->at(0).getBatchSize();

  uint64_t num_test_batches = std::min(test_data->numBatches(), batch_limit);

  MetricAggregator metrics(metric_names, verbose);

  // Because of how the datasets are read we know that all batches will not have
  // a batch size larger than this so we can just set the batch size here.
  // If sparse inference is not enabled we want the outptus to be dense,
  // otherwise we want whatever the default for the layer is.
  prepareForBatchProcessing(batch_size, /* use_sparsity= */ false);

  ProgressBar bar(num_test_batches, verbose);

  auto test_start = std::chrono::high_resolution_clock::now();
  for (uint32_t batch = 0; batch < num_test_batches; batch++) {
    BATCH_T& inputs = test_data->at(batch);

    const BoltBatch* batch_labels =
        compute_metrics ? &(*test_labels)[batch] : nullptr;

    processInferenceBatch(inputs, batch_labels, metrics, compute_metrics);

    bar.increment();
  }

  auto test_end = std::chrono::high_resolution_clock::now();
  int64_t test_time = std::chrono::duration_cast<std::chrono::milliseconds>(
                          test_end - test_start)
                          .count();
  if (verbose) {
    std::cout << std::endl
              << "Processed " << num_test_batches << " test batches in "
              << test_time << " milliseconds" << std::endl;
  }

  metrics.logAndReset();
  auto metric_vals = metrics.getOutputFromInference();
  metric_vals["test_time"] = test_time;

  return metric_vals;
}

template <>
void BoltGraph::processInferenceBatch(BoltBatch& batch_inputs,
                                      const BoltBatch* batch_labels,
                                      MetricAggregator& metrics,
                                      bool compute_metrics) {
  _inputs[0]->setInputs(&batch_inputs);

#pragma omp parallel for default(none) \
    shared(batch_inputs, batch_labels, metrics, compute_metrics)
  for (uint32_t vec_id = 0; vec_id < batch_inputs.getBatchSize(); vec_id++) {
    // We set labels to nullptr so that they are not used in sampling during
    // inference.
    forward(vec_id, /*labels=*/nullptr);

    if (compute_metrics) {
      metrics.processSample(_output->getOutput(vec_id),
                            (*batch_labels)[vec_id]);
    }
  }
}

void BoltGraph::forward(uint32_t batch_index, const BoltVector* labels) {
  for (uint32_t i = 0; i < _nodes.size() - 1; i++) {
    _nodes[i]->forward(batch_index, nullptr);
  }
  _nodes.back()->forward(batch_index, labels);
}

void BoltGraph::backpropagate(uint32_t batch_index) {
  for (uint32_t i = _nodes.size(); i > 0; i--) {
    _nodes[i - 1]->backpropagate(batch_index);
  }
}

void BoltGraph::prepareForBatchProcessing(uint32_t batch_size,
                                          bool use_sparsity) {
  for (auto& node : _nodes) {
    node->prepareForBatchProcessing(batch_size, use_sparsity);
  }
}

void BoltGraph::updateParameters(float learning_rate, uint32_t batch_cnt) {
  for (auto& node : _nodes) {
    node->updateParameters(learning_rate, batch_cnt);
  }
}

void BoltGraph::traverseGraph() {
  std::queue<NodePtr> queue;
  std::unordered_set<NodePtr> visited;

  queue.push(_output);

  while (!queue.empty()) {
    auto& next = queue.front();
    if (!visited.count(next) && !next->isInputNode()) {
      _nodes.push_back(next);

      auto predecessors = next->getPredecessors();
      for (auto& pred : predecessors) {
        queue.push(std::move(pred));
      }
    }
    queue.pop();
  }

  std::reverse(_nodes.begin(), _nodes.end());
}

void BoltGraph::rebuildHashTables() {
  for (auto& layer : _sparse_layers) {
    layer->buildHashTables();
  }
}

void BoltGraph::rebuildHashFunctions() {
  for (auto& layer : _sparse_layers) {
    layer->reBuildHashFunction();
  }
}

}  // namespace thirdai::bolt