#include "Graph.h"
#include <cereal/types/memory.hpp>
#include <cereal/types/optional.hpp>
#include <cereal/types/vector.hpp>
#include <cereal/archives/binary.hpp>
#include "GraphPropertyChecks.h"
#include "nodes/FullyConnected.h"
#include <bolt/src/graph/Node.h>
#include <bolt/src/graph/nodes/Input.h>
#include <bolt/src/layers/BoltVector.h>
#include <bolt/src/loss_functions/LossFunctions.h>
#include <bolt/src/metrics/MetricAggregator.h>
#include <bolt/src/utils/ProgressBar.h>
#include <exceptions/src/Exceptions.h>
#include <algorithm>
#include <chrono>
#include <exception>
#include <optional>
#include <ostream>
#include <queue>
#include <stdexcept>
#include <type_traits>
#include <unordered_set>

namespace thirdai::bolt {

void BoltGraph::compile(std::shared_ptr<LossFunction> loss,
                        bool print_when_done) {
  if (_output == nullptr) {
    throw exceptions::GraphCompilationFailure(
        "Output NodePtr cannot be a nullptr.");
  }

  _loss = std::move(loss);

  verifyGraphProperties();

  traverseGraph();

  LayerNameManager name_manager;
  for (auto& node : getNodeTraversalOrder()) {
    node->compile(name_manager);
  }

  std::unordered_map<std::string, uint32_t> layer_type_name_to_count;
  for (auto& node : _nodes) {
    auto node_layers = node->getInternalFullyConnectedLayers();
    _internal_fully_connected_layers.insert(
        _internal_fully_connected_layers.end(), node_layers.begin(),
        node_layers.end());
  }

  if (print_when_done) {
    summarize(/* print = */ true, /* detailed = */ false);
  }
}

template MetricData BoltGraph::train(
    std::shared_ptr<dataset::InMemoryDataset<BoltBatch>>&,
    const dataset::BoltDatasetPtr&, const TrainConfig& train_config);

template MetricData BoltGraph::train(
    std::shared_ptr<dataset::InMemoryDataset<dataset::BoltTokenBatch>>&,
    const dataset::BoltDatasetPtr&, const TrainConfig& train_config);

template <typename BATCH_T>
MetricData BoltGraph::train(
    std::shared_ptr<dataset::InMemoryDataset<BATCH_T>>& train_data,
    const dataset::BoltDatasetPtr& train_labels,
    const TrainConfig& train_config) {
  verifyInputForGraph(train_data);
  if (!graphCompiled()) {
    throw std::logic_error("Graph must be compiled before training");
  }

  // TODO(Nicholas): Switch to batch_size property in dataset.
  uint32_t max_batch_size = train_data->at(0).getBatchSize();
  if (max_batch_size == 0) {
    throw std::invalid_argument("Batch size must be greater than 0");
  }

  uint32_t rebuild_hash_tables_batch =
      train_config.getRebuildHashTablesBatchInterval(max_batch_size,
                                                     train_data->len());

  uint32_t reconstruct_hash_functions_batch =
      train_config.getReconstructHashFunctionsBatchInterval(max_batch_size,
                                                            train_data->len());

  uint64_t num_train_batches = train_data->numBatches();

  /*
    Because of how the datasets are read we know that all batches will not have
    a batch size larger than the first batch_size. We will be using the same
    datastructures to store the activations for every batch during training so
    we need this to be able to support the largest batch size.
   */
  prepareToProcessBatches(max_batch_size, /* use_sparsity=*/true);

  std::vector<double> time_per_epoch;

  MetricAggregator metrics = train_config.getMetricAggregator();

  for (uint32_t epoch = 0; epoch < train_config.epochs(); epoch++) {
    if (train_config.verbose()) {
      std::cout << "\nEpoch " << (_epoch_count + 1) << ':' << std::endl;
    }
    ProgressBar bar(num_train_batches, train_config.verbose());
    auto train_start = std::chrono::high_resolution_clock::now();

    for (uint32_t batch = 0; batch < num_train_batches; batch++) {
      BATCH_T& batch_inputs = train_data->at(batch);

      const BoltBatch& batch_labels = train_labels->at(batch);

      processTrainingBatch(batch_inputs, batch_labels,
                           train_config.learningRate(), metrics);

      updateSampling(/* rebuild_hash_tables_batch= */ rebuild_hash_tables_batch,
                     /* reconstruct_hash_functions_batch= */
                     reconstruct_hash_functions_batch);

      bar.increment();
    }

    auto train_end = std::chrono::high_resolution_clock::now();
    int64_t epoch_time = std::chrono::duration_cast<std::chrono::seconds>(
                             train_end - train_start)
                             .count();

    time_per_epoch.push_back(static_cast<double>(epoch_time));
    if (train_config.verbose()) {
      std::cout << std::endl
                << "Processed " << num_train_batches << " training batches in "
                << epoch_time << " seconds" << std::endl;
    }
    _epoch_count++;
    metrics.logAndReset();
  }

  cleanupAfterBatchProcessing();

  auto metric_data = metrics.getOutput();
  metric_data["epoch_times"] = std::move(time_per_epoch);

  return metric_data;
}

template <typename BATCH_T>
void BoltGraph::processTrainingBatch(BATCH_T& batch_inputs,
                                     const BoltBatch& batch_labels,
                                     float learning_rate,
                                     MetricAggregator& metrics) {
  assert(graphCompiled());
  setInputs(batch_inputs);

#pragma omp parallel for default(none) \
    shared(batch_inputs, batch_labels, metrics)
  for (uint32_t vec_id = 0; vec_id < batch_inputs.getBatchSize(); vec_id++) {
    forward(vec_id, &batch_labels[vec_id]);

    _loss->lossGradients(_output->getOutputVector(vec_id), batch_labels[vec_id],
                         batch_inputs.getBatchSize());

    backpropagate(vec_id);

    metrics.processSample(_output->getOutputVector(vec_id),
                          batch_labels[vec_id]);
  }

  ++_batch_cnt;
  updateParameters(learning_rate, _batch_cnt);
}

void BoltGraph::updateSampling(uint32_t rebuild_hash_tables_batch,
                               uint32_t reconstruct_hash_functions_batch) {
  if (checkBatchInterval(reconstruct_hash_functions_batch)) {
    reconstructHashFunctions();
    rebuildHashTables();
  } else if (checkBatchInterval(rebuild_hash_tables_batch)) {
    rebuildHashTables();
  }
}

template InferenceResult BoltGraph::predict(
    const std::shared_ptr<dataset::InMemoryDataset<BoltBatch>>&,
    const dataset::BoltDatasetPtr&, const PredictConfig&);

template InferenceResult BoltGraph::predict(
    const std::shared_ptr<dataset::InMemoryDataset<dataset::BoltTokenBatch>>&,
    const dataset::BoltDatasetPtr&, const PredictConfig&);

template <typename BATCH_T>
InferenceResult BoltGraph::predict(
    // Test dataset
    const std::shared_ptr<dataset::InMemoryDataset<BATCH_T>>& test_data,
    // Test labels
    const dataset::BoltDatasetPtr& test_labels,
    // Other prediction parameters
    const PredictConfig& predict_config) {
  bool has_labels = (test_labels != nullptr);
  MetricAggregator metrics = predict_config.getMetricAggregator();

  verifyCanPredict(
      test_data, /* has_labels = */ has_labels,
      /* returning_activations = */ predict_config.shouldReturnActivations(),
      /* num_metrics_tracked = */ metrics.getNumMetricsTracked());

  uint32_t max_batch_size = test_data->at(0).getBatchSize();
  uint64_t num_test_batches = test_data->numBatches();

  /*
   Because of how the datasets are read we know that all batches will not have
   a batch size larger than the first batch_size. We will be using the same
   datastructures to store the activations for every batch during training so
   we need this to be able to support the largest batch size.
  */
  prepareToProcessBatches(max_batch_size,
                          predict_config.sparseInferenceEnabled());

  InferenceOutputTracker outputTracker(
      _output, predict_config, /* total_num_samples = */ test_data->len());

  ProgressBar bar(num_test_batches, predict_config.verbose());

  auto test_start = std::chrono::high_resolution_clock::now();
  for (uint32_t batch = 0; batch < num_test_batches; batch++) {
    BATCH_T& inputs = test_data->at(batch);

    const BoltBatch* batch_labels =
        has_labels ? &(*test_labels)[batch] : nullptr;

    processInferenceBatch(inputs, batch_labels, metrics);

    bar.increment();

    outputTracker.saveOutputBatch(_output, inputs.getBatchSize());
  }

  cleanupAfterBatchProcessing();

  auto test_end = std::chrono::high_resolution_clock::now();
  int64_t test_time = std::chrono::duration_cast<std::chrono::milliseconds>(
                          test_end - test_start)
                          .count();

  if (predict_config.verbose()) {
    std::cout << std::endl
              << "Processed " << num_test_batches << " test batches in "
              << test_time << " milliseconds" << std::endl;
  }

  metrics.logAndReset();
  auto metric_vals = metrics.getOutputFromInference();
  metric_vals["test_time"] = test_time;

  return {std::move(metric_vals), std::move(outputTracker)};
}

template <typename BATCH_T>
void BoltGraph::processInferenceBatch(BATCH_T& batch_inputs,
                                      const BoltBatch* batch_labels,
                                      MetricAggregator& metrics) {
  setInputs(batch_inputs);

  // Either we shouldn't track any metrics or there need to be labels
  assert((metrics.getNumMetricsTracked() == 0) || (batch_labels != nullptr));

#pragma omp parallel for default(none) \
    shared(batch_inputs, batch_labels, metrics)
  for (uint32_t vec_id = 0; vec_id < batch_inputs.getBatchSize(); vec_id++) {
    // We set labels to nullptr so that they are not used in sampling during
    // inference.
    forward(vec_id, /*labels=*/nullptr);

    const auto& output = _output->getOutputVector(vec_id);

    if (batch_labels) {
      const auto& labels = (*batch_labels)[vec_id];
      metrics.processSample(output, labels);
    }
  }
}

// This syntax means that we are implmenting the function for the specific case
// in which BATCH_T is equivalent to BoltBatch.
template <>
void BoltGraph::setInputs(BoltBatch& batch_inputs) {
  // If we are using a BoltBatch then there is only one input. This is
  // checked in the verifyInputForGraph() function.
  _inputs[0]->setInputs(&batch_inputs);
}

// This syntax means that we are implmenting the function for the specific case
// in which BATCH_T is equivalent to BoltTokenBatch.
template <>
void BoltGraph::setInputs(dataset::BoltTokenBatch& batch_inputs) {
  // If we are using a BoltTokenBatch then there is only one token input. This
  // is checked in the verifyInputForGraph() function.
  _token_inputs[0]->setTokenInputs(&batch_inputs);
}

void BoltGraph::forward(uint32_t vec_index, const BoltVector* labels) {
  for (uint32_t i = 0; i < _nodes.size() - 1; i++) {
    _nodes[i]->forward(vec_index, nullptr);
  }
  _nodes.back()->forward(vec_index, labels);
}

void BoltGraph::backpropagate(uint32_t vec_index) {
  for (auto node_itr = _nodes.rbegin(); node_itr != _nodes.rend(); ++node_itr) {
    (*node_itr)->backpropagate(vec_index);
  }
}

void BoltGraph::prepareToProcessBatches(uint32_t batch_size,
                                        bool use_sparsity) {
  for (auto& node : _nodes) {
    node->prepareForBatchProcessing(batch_size, use_sparsity);
  }
}

void BoltGraph::cleanupAfterBatchProcessing() {
  for (auto& node : _nodes) {
    node->cleanupAfterBatchProcessing();
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

  std::unordered_set<NodePtr> all_inputs;
  all_inputs.insert(_inputs.begin(), _inputs.end());
  all_inputs.insert(_token_inputs.begin(), _token_inputs.end());

  std::unordered_map<NodePtr, int32_t> successor_counts = getSuccessorCounts();

  queue.push(_output);

  while (!queue.empty()) {
    auto& next = queue.front();
    if (!visited.count(next) && !next->isInputNode()) {
      _nodes.push_back(next);
      visited.insert(next);

      auto predecessors = next->getPredecessors();
      for (auto& pred : predecessors) {
        successor_counts[pred]--;
        if (successor_counts[pred] == 0) {
          queue.push(pred);
        }
      }
    }
    if (next->isInputNode()) {
      if (!all_inputs.count(next)) {
        throw exceptions::GraphCompilationFailure(
            "Found input that was not provided in list of input nodes.");
      }
      all_inputs.erase(next);
    }
    queue.pop();
  }

  if (!all_inputs.empty()) {
    throw exceptions::GraphCompilationFailure(
        "Not all provided inputs were reached during graph traversal.");
  }

  for (auto [node, cnt] : successor_counts) {
    if (cnt != 0) {
      throw exceptions::GraphCompilationFailure(
          "Cannot compile model from graph containing a cycle.");
    }
  }

  std::reverse(_nodes.begin(), _nodes.end());
}

std::unordered_map<NodePtr, int32_t> BoltGraph::getSuccessorCounts() const {
  std::unordered_map<NodePtr, int32_t> num_successors;

  std::queue<NodePtr> queue;
  std::unordered_set<NodePtr> visited;

  queue.push(_output);

  while (!queue.empty()) {
    auto& next = queue.front();
    if (!visited.count(next)) {
      visited.insert(next);

      auto predecessors = next->getPredecessors();
      for (auto& pred : predecessors) {
        num_successors[pred]++;
        queue.push(pred);
      }
    }

    queue.pop();
  }

  if (num_successors[_output] != 0) {
    throw exceptions::GraphCompilationFailure(
        "Output node cannot have successor nodes.");
  }

  return num_successors;
}

template <typename BATCH_T>
void BoltGraph::verifyCanPredict(
    const std::shared_ptr<dataset::InMemoryDataset<BATCH_T>>& test_data,
    bool has_labels, bool returning_activations, uint32_t num_metrics_tracked) {
  verifyInputForGraph(test_data);

  if (!graphCompiled()) {
    throw std::logic_error("Graph must be compiled before inference");
  }

  if (!has_labels && num_metrics_tracked != 0) {
    throw std::invalid_argument("Cannot track accuracy metrics without labels");
  }
  if (!returning_activations && num_metrics_tracked == 0) {
    throw std::invalid_argument(
        "Doing inference without returning activations and no metrics is a "
        "NOOP");
  }
}

template <typename BATCH_T>
void BoltGraph::verifyInputForGraph(
    const std::shared_ptr<dataset::InMemoryDataset<BATCH_T>>& dataset) {
  (void)dataset;
  if (std::is_same<BATCH_T, BoltBatch>::value && _inputs.size() != 1 &&
      !_token_inputs.empty()) {
    throw exceptions::GraphCompilationFailure(
        "Only graphs with a single input layer can take in a dataset with "
        "batch type BoltBatch.");
  }

  if (std::is_same<BATCH_T, dataset::BoltTokenBatch>::value &&
      !_inputs.empty() && _token_inputs.size() != 1) {
    throw exceptions::GraphCompilationFailure(
        "Only graphs with a single token input layer can take in a dataset "
        "with batch type BoltTokenBatch.");
  }
}

void BoltGraph::verifyGraphProperties() {
  GraphPropertyChecks::verifyOutputIsNotInputLayer(_output);

  GraphPropertyChecks::verifyOutputIsNotConcatLayer(_output);

  GraphPropertyChecks::verifySoftmaxIsUsedWithCategoricalCrossEntropy(_output,
                                                                      _loss);

  GraphPropertyChecks::verifySigmoidIsUsedWithBinaryCrossEntropy(_output,
                                                                 _loss);
}

void BoltGraph::rebuildHashTables() {
  for (auto& layer : _internal_fully_connected_layers) {
    layer->buildHashTables();
  }
}

void BoltGraph::reconstructHashFunctions() {
  for (auto& layer : _internal_fully_connected_layers) {
    layer->reBuildHashFunction();
  }
}

void BoltGraph::freezeHashTables(bool insert_labels_if_not_found) {
  for (auto& layer : _internal_fully_connected_layers) {
    layer->freezeHashTables(/* insert_labels_if_not_found= */ false);
  }

  if (insert_labels_if_not_found) {
    for (auto& layer : _output->getInternalFullyConnectedLayers()) {
      layer->freezeHashTables(/* insert_labels_if_not_found= */ true);
    }
  }
}

template void BoltGraph::serialize(cereal::BinaryOutputArchive& archive);
template <class Archive>
void BoltGraph::serialize(Archive& archive) {
  archive(_nodes, _output, _inputs, _internal_fully_connected_layers, _loss,
          _epoch_count, _batch_cnt);
}

void BoltGraph::save(const std::string& filename) {
  if (!graphCompiled()) {
    throw exceptions::NodeStateMachineError(
        "Cannot save graph that is not compiled.");
  }
  std::ofstream filestream =
      dataset::SafeFileIO::ofstream(filename, std::ios::binary);
  cereal::BinaryOutputArchive oarchive(filestream);
  oarchive(*this);
}

std::unique_ptr<BoltGraph> BoltGraph::load(const std::string& filename) {
  std::ifstream filestream =
      dataset::SafeFileIO::ifstream(filename, std::ios::binary);
  cereal::BinaryInputArchive iarchive(filestream);
  std::unique_ptr<BoltGraph> deserialize_into(new BoltGraph());
  iarchive(*deserialize_into);
  return deserialize_into;
}

std::string BoltGraph::summarize(bool print, bool detailed) const {
  if (!graphCompiled()) {
    throw std::logic_error("Cannot summarize the graph before it is compiled.");
  }
  std::stringstream summary;
  summary << "\n";
  summary << "======================= Bolt Model =======================\n";
  for (const auto& node : getNodeTraversalOrder()) {
    node->summarize(summary, detailed);
  }
  summary << "============================================================\n";
  if (print) {
    std::cout << summary.str() << std::flush;
  }
  return summary.str();
}

NodePtr BoltGraph::getNodeByName(const std::string& node_name) const {
  if (!graphCompiled()) {
    throw std::logic_error(
        "Cannot get a node by name from the graph before it is compiled.");
  }
  for (const auto& node : getNodeTraversalOrder()) {
    if (node->name() == node_name) {
      return node;
    }
  }
  throw std::invalid_argument("A node with name \"" + node_name +
                              "\" was not found");
}

}  // namespace thirdai::bolt