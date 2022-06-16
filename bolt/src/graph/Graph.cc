#include "Graph.h"
#include <bolt/src/graph/nodes/Input.h>
#include <bolt/src/layers/BoltVector.h>
#include <bolt/src/utils/ProgressBar.h>
#include <queue>
#include <unordered_set>

namespace thirdai::bolt {

void BoltGraph::compile(std::shared_ptr<LossFunction> loss) {
  _loss = std::move(loss);

  for (auto& node : _nodes) {
    node->compile();
  }
}

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
  initializeState(batch_size, true);

  std::vector<double> time_per_epoch;

  MetricAggregator metrics(metric_names, verbose);

  for (uint32_t epoch = 0; epoch < epochs; epoch++) {
    if (verbose) {
      std::cout << "\nEpoch " << (_epoch_count + 1) << ':' << std::endl;
    }
    ProgressBar bar(num_train_batches, verbose);
    auto train_start = std::chrono::high_resolution_clock::now();

    for (uint32_t batch = 0; batch < num_train_batches; batch++) {
      if (_batch_cnt % 1000 == 999) {
        shuffleRandomNeurons();
      }

      BATCH_T& batch_inputs = train_data->at(batch);

      const BoltBatch& batch_labels = train_labels->at(batch);

      processTrainingBatch(batch_inputs, batch_labels, learning_rate,
                           rehash_batch, rebuild_batch, metrics);

      if (_batch_cnt % rebuild_batch == (rebuild_batch - 1)) {
        rebuildHashFunctions();
        rebuildHashTables();
      } else if (_batch_cnt % rehash_batch == (rehash_batch - 1)) {
        rebuildHashTables();
      }

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
void BoltGraph::processTrainingBatch(BoltBatch& batch_data,
                                     const BoltBatch& batch_labels,
                                     float learning_rate,
                                     MetricAggregator& metrics) {
  _inputs[0]->setInputs(&batch_data);

#pragma omp parallel for default(none) shared(batch_data, batch_labels, metrics)
  for (uint32_t vec_id = 0; vec_id < batch_data.getBatchSize(); vec_id++) {
    forward(vec_id, &batch_labels[vec_id]);

    _loss->lossGradients(_output->getOutput(vec_id), batch_labels[vec_id],
                         batch_data.getBatchSize());

    backpropagate(vec_id);

    metrics.processSample(_output->getOutput(vec_id), batch_labels[vec_id]);
  }

  updateParameters(learning_rate, ++_batch_cnt);
}

void BoltGraph::forward(uint32_t batch_index, const BoltVector* labels) {
  for (uint32_t i = 0; i < _nodes.size() - 1; i++) {
    _nodes[i]->forward(batch_index, nullptr);
  }
  _nodes.back()->forward(batch_index, labels);
}

void BoltGraph::backpropagate(uint32_t batch_index) {
  for (uint32_t i = _nodes.size() - 1; i >= 0; i--) {
    _nodes[i]->backpropagate(batch_index);
  }
}

void BoltGraph::initializeState(uint32_t batch_size, bool use_sparsity) {
  for (auto& node : _nodes) {
    node->initializeState(batch_size, use_sparsity);
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
  std::vector<NodePtr> reverse_order_layers;

  queue.push(_output);

  while (!queue.empty()) {
    auto& next = queue.front();
    if (!visited.count(next) && !next->isInputNode()) {
      reverse_order_layers.push_back(next);
      next->enqueuePredecessors(queue);
    }
    queue.pop();
  }

  for (auto& node : reverse_order_layers) {
    _nodes.push_back(node);
  }
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

void BoltGraph::shuffleRandomNeurons() {
  for (auto& layer : _sparse_layers) {
    layer->shuffleRandNeurons();
  }
}

}  // namespace thirdai::bolt