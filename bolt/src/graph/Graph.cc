#include "Graph.h"
#include <cereal/types/memory.hpp>
#include <cereal/types/optional.hpp>
#include <cereal/types/vector.hpp>
#include "GraphPropertyChecks.h"
#include "nodes/FullyConnected.h"
#include <bolt/src/graph/DatasetContext.h>
#include <bolt/src/graph/Node.h>
#include <bolt/src/graph/nodes/Input.h>
#include <bolt/src/layers/BoltVector.h>
#include <bolt/src/loss_functions/LossFunctions.h>
#include <bolt/src/metrics/MetricAggregator.h>
#include <bolt/src/utils/ProgressBar.h>
#include <exceptions/src/Exceptions.h>
#include <algorithm>
#include <chrono>
#include <csignal>
#include <exception>
#include <optional>
#include <ostream>
#include <queue>
#include <stdexcept>
#include <string>
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

MetricData BoltGraph::train(
    const std::vector<dataset::BoltDatasetPtr>& train_data,
    const std::vector<dataset::BoltTokenDatasetPtr>& train_tokens,
    const dataset::BoltDatasetPtr& train_labels,
    const TrainConfig& train_config) {
  DatasetContext train_context(train_data, train_tokens, train_labels);

  verifyCanTrain(train_context);

  uint32_t rebuild_hash_tables_batch =
      train_config.getRebuildHashTablesBatchInterval(train_context.batchSize(),
                                                     train_context.len());

  uint32_t reconstruct_hash_functions_batch =
      train_config.getReconstructHashFunctionsBatchInterval(
          train_context.batchSize(), train_context.len());

  /*
    Because of how the datasets are read we know that all batches will not have
    a batch size larger than the first batch_size. We will be using the same
    datastructures to store the activations for every batch during training so
    we need this to be able to support the largest batch size.
   */
  prepareToProcessBatches(train_context.batchSize(), /* use_sparsity=*/true);

  std::vector<double> time_per_epoch;

  MetricAggregator metrics = train_config.getMetricAggregator();

  // TODO(josh/Nick): This try catch is kind of a hack, we should really use
  // some sort of RAII training context object whose destructor will
  // automatically delete the training state
  try {
    for (uint32_t epoch = 0; epoch < train_config.epochs(); epoch++) {
      if (train_config.verbose()) {
        std::cout << "\nEpoch " << (_epoch_count + 1) << ':' << std::endl;
      }
      ProgressBar bar(train_context.numBatches(), train_config.verbose());
      auto train_start = std::chrono::high_resolution_clock::now();

      for (uint64_t batch_idx = 0; batch_idx < train_context.numBatches();
           batch_idx++) {
        train_context.setInputs(batch_idx, _inputs, _token_inputs);

        const BoltBatch& batch_labels = train_context.labels()->at(batch_idx);
        processTrainingBatch(batch_labels, train_config.learningRate(),
                             metrics);

        updateSampling(
            /* rebuild_hash_tables_batch= */ rebuild_hash_tables_batch,
            /* reconstruct_hash_functions_batch= */
            reconstruct_hash_functions_batch);

        bar.increment();
      }

      perEpochCallback();

      auto train_end = std::chrono::high_resolution_clock::now();
      int64_t epoch_time = std::chrono::duration_cast<std::chrono::seconds>(
                               train_end - train_start)
                               .count();

      time_per_epoch.push_back(static_cast<double>(epoch_time));
      if (train_config.verbose()) {
        std::cout << std::endl
                  << "Processed " << train_context.numBatches()
                  << " training batches in " << epoch_time << " seconds"
                  << std::endl;
      }
      _epoch_count++;
      metrics.logAndReset();
    }
  } catch (const std::exception& e) {
    cleanupAfterBatchProcessing();
    throw;
  }

  cleanupAfterBatchProcessing();

  auto metric_data = metrics.getOutput();
  metric_data["epoch_times"] = std::move(time_per_epoch);

  return metric_data;
}

void BoltGraph::processTrainingBatch(const BoltBatch& batch_labels,
                                     float learning_rate,
                                     MetricAggregator& metrics) {
  assert(graphCompiled());
  batch_labels.verifyExpectedDimension(
      /* expected_dimension = */ _output->outputDim(),
      /* origin_string = */
      "Passed in label BoltVector is larger than the output dim");

#pragma omp parallel for default(none) shared(batch_labels, metrics)
  for (uint64_t vec_id = 0; vec_id < batch_labels.getBatchSize(); vec_id++) {
    forward(vec_id, &batch_labels[vec_id]);

    _loss->lossGradients(_output->getOutputVector(vec_id), batch_labels[vec_id],
                         batch_labels.getBatchSize());

    backpropagate(vec_id);

    metrics.processSample(_output->getOutputVector(vec_id),
                          batch_labels[vec_id]);
  }

  perBatchCallback();

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

BoltVector BoltGraph::getLabelVectorExplainPrediction(uint32_t vec_id,
                                                      bool explain_prediction) {
  uint32_t required_index;
  forward(vec_id, nullptr);
  if (explain_prediction) {
    required_index = _output->getOutputVector(vec_id).getHighestActivationId();
  } else {
    required_index =
        _output->getOutputVector(vec_id).getSecondHighestActivationId();
  }
  return BoltVector::makeSparseVector({required_index}, {1.0});
}

BoltVector BoltGraph::getLabelVectorNeuronsToExplain(uint32_t required_index,
                                                     uint32_t vec_id) {
  if (required_index >= _output->outputDim()) {
    throw std::invalid_argument(
        "Cannot pass required index " + std::to_string(required_index) +
        " to getInputGradients for network with output dim " +
        std::to_string(_output->outputDim()));
  }
  BoltVector label_vector =
      BoltVector::makeSparseVector({required_index}, {1.0});
  forward(vec_id, &label_vector);
  return label_vector;
}

std::pair<std::optional<std::vector<uint32_t>>, std::vector<float>>
BoltGraph::getInputGradientSingle(std::vector<BoltVector>&& input_data,
                                  bool explain_prediction, bool neuron_given,
                                  uint32_t neuron_to_explain) {
  SingleUnitDatasetContext single_input_gradients_context(std::move(input_data),
                                                          {});

  verifyCanGetInputGradientSingle(single_input_gradients_context,
                                  explain_prediction,
                                  _output->numNonzerosInOutput());

  prepareToProcessBatches(1, /* use_sparsity=*/true);

  try {
    single_input_gradients_context.setInputs(/* batch_idx = */ 0, _inputs,
                                             _token_inputs);
    std::vector<float> vec_grad(_inputs[0]->getOutputVector(0).len, 0.0);
    _inputs[0]->getOutputVector(0).gradients = vec_grad.data();
    std::vector<uint32_t> input_dataset_indices;
    BoltVector label_vector;
    if (!neuron_given) {
      label_vector = getLabelVectorExplainPrediction(0, explain_prediction);
    } else {
      label_vector = getLabelVectorNeuronsToExplain(neuron_to_explain, 0);
    }
    if (!_inputs[0]->getOutputVector(0).isDense()) {
      input_dataset_indices.assign(
          _inputs[0]->getOutputVector(0).active_neurons,
          _inputs[0]->getOutputVector(0).active_neurons +
              _inputs[0]->getOutputVector(0).len);
    }
    _loss->lossGradients(_output->getOutputVector(0), label_vector, 1);
    backpropagate(0);
    _inputs[0]->getOutputVector(0).gradients = nullptr;
    cleanupAfterBatchProcessing();

    if (input_dataset_indices.empty()) {
      return std::make_pair(std::nullopt, vec_grad);
    }
    return std::make_pair(input_dataset_indices, vec_grad);
  } catch (const std::exception& e) {
    cleanupAfterBatchProcessing();
    throw;
  }
}

// TODO (YASH) : ( Extend this getInputGradients for multiple inputs.)
std::pair<std::optional<std::vector<std::vector<uint32_t>>>,
          std::vector<std::vector<float>>>
BoltGraph::getInputGradients(const dataset::BoltDatasetPtr& input_data,
                             bool explain_prediction,
                             const std::vector<uint32_t>& neurons_to_explain) {
  DatasetContext input_gradients_context({input_data}, {}, nullptr);

  // Because of how the datasets are read we know that all batches will not
  // have a batch size larger than this so we can just set the batch size
  // here.

  prepareToProcessBatches(input_gradients_context.batchSize(),
                          /* use_sparsity=*/true);

  verifyCanGetInputGradients(input_gradients_context, neurons_to_explain.size(),
                             input_data->len(), explain_prediction,
                             _output->numNonzerosInOutput());

  std::vector<std::vector<float>> input_dataset_grad;
  std::vector<std::vector<uint32_t>> input_dataset_indices;
  try {
    for (uint64_t batch_idx = 0;
         batch_idx < input_gradients_context.numBatches(); batch_idx++) {
      input_gradients_context.setInputs(batch_idx, _inputs, _token_inputs);
      for (uint32_t vec_id = 0;
           vec_id < input_gradients_context.batchSize(batch_idx); vec_id++) {
        std::vector<float> vec_grad(_inputs[0]->getOutputVector(vec_id).len,
                                    0.0);
        // Assigning the vec_grad data() to gradients so that we dont have to
        // worry about initializing and then freeing the memory.
        _inputs[0]->getOutputVector(vec_id).gradients = vec_grad.data();
        /*
        If the required_labels are empty, then we have to find the
        required_index by output activations, for that we need to do forward
        pass before creating the batch_label, but if the required_labels are not
        empty and for some ,If the required label position is not present in the
        output active neurons , then calculating the gradients with respect to
        that label doesnot make sense, because loss is only calculated with
        respect to active neurons, to ensure that output has active neuron at
        the position of required label we are creating batch_label before
        forward pass and passing to it, because forward pass ensures to have
        active neurons at the metioned label index.
        */
        BoltVector label_vector;
        if (neurons_to_explain.empty()) {
          label_vector =
              getLabelVectorExplainPrediction(vec_id, explain_prediction);
        } else {
          uint32_t required_index =
              neurons_to_explain[batch_idx *
                                     input_gradients_context.batchSize(0) +
                                 vec_id];
          label_vector = getLabelVectorNeuronsToExplain(required_index, vec_id);
        }
        if (!_inputs[0]->getOutputVector(vec_id).isDense()) {
          std::vector<uint32_t> vec_indices(
              _inputs[0]->getOutputVector(vec_id).active_neurons,
              _inputs[0]->getOutputVector(vec_id).active_neurons +
                  _inputs[0]->getOutputVector(vec_id).len);
          input_dataset_indices.push_back(vec_indices);
        }
        _loss->lossGradients(_output->getOutputVector(vec_id), label_vector,
                             input_gradients_context.batchSize(batch_idx));
        backpropagate(vec_id);

        // We reset the gradients to nullptr here to prevent the bolt vector
        // from freeing the memory which is owned by the std::vector we used to
        // store the gradients

        _inputs[0]->getOutputVector(vec_id).gradients = nullptr;
        input_dataset_grad.push_back(vec_grad);
      }
    }
  } catch (const std::exception& e) {
    cleanupAfterBatchProcessing();
    throw;
  }
  cleanupAfterBatchProcessing();

  if (input_dataset_indices.empty()) {
    return std::make_pair(std::nullopt, input_dataset_grad);
  }
  return std::make_pair(input_dataset_indices, input_dataset_grad);
}

InferenceResult BoltGraph::predict(
    const std::vector<dataset::BoltDatasetPtr>& test_data,
    const std::vector<dataset::BoltTokenDatasetPtr>& test_tokens,
    const dataset::BoltDatasetPtr& test_labels,
    const PredictConfig& predict_config) {
  DatasetContext predict_context(test_data, test_tokens, test_labels);

  bool has_labels = (test_labels != nullptr);

  MetricAggregator metrics = predict_config.getMetricAggregator();

  verifyCanPredict(
      predict_context, has_labels,
      /* returning_activations = */ predict_config.shouldReturnActivations(),
      /* num_metrics_tracked = */ metrics.getNumMetricsTracked());

  /*
   Because of how the datasets are read we know that all batches will not have
   a batch size larger than the first batch_size. We will be using the same
   datastructures to store the activations for every batch during training so
   we need this to be able to support the largest batch size.
  */
  prepareToProcessBatches(predict_context.batchSize(),
                          predict_config.sparseInferenceEnabled());

  InferenceOutputTracker outputTracker(
      _output, predict_config.shouldReturnActivations(),
      /* total_num_samples = */ predict_context.len());

  ProgressBar bar(predict_context.numBatches(), predict_config.verbose());

  auto test_start = std::chrono::high_resolution_clock::now();

  // TODO(josh/Nick): This try catch is kind of a hack, we should really use
  // some sort of RAII training context object whose destructor will
  // automatically delete the training state
  try {
    for (uint64_t batch_idx = 0; batch_idx < predict_context.numBatches();
         batch_idx++) {
      predict_context.setInputs(batch_idx, _inputs, _token_inputs);

      uint64_t batch_size = predict_context.batchSize(batch_idx);
      const BoltBatch* batch_labels =
          has_labels ? &predict_context.labels()->at(batch_idx) : nullptr;

      processInferenceBatch(batch_size, batch_labels, metrics);

      bar.increment();

      processOutputCallback(predict_config.outputCallback(), batch_size);

      outputTracker.saveOutputBatch(_output, batch_size);
    }
  } catch (const std::exception& e) {
    cleanupAfterBatchProcessing();
    throw;
  }

  cleanupAfterBatchProcessing();

  auto test_end = std::chrono::high_resolution_clock::now();
  int64_t test_time = std::chrono::duration_cast<std::chrono::milliseconds>(
                          test_end - test_start)
                          .count();

  if (predict_config.verbose()) {
    std::cout << std::endl
              << "Processed " << predict_context.numBatches()
              << " test batches in " << test_time << " milliseconds"
              << std::endl;
  }

  metrics.logAndReset();
  auto metric_vals = metrics.getOutputFromInference();
  metric_vals["test_time"] = test_time;

  return {std::move(metric_vals), std::move(outputTracker)};
}

// Predicts on a single sample input for performance. Always returns
// activations and doesn't calculate metrics.
BoltVector BoltGraph::predictSingle(
    std::vector<BoltVector>&& test_data,
    std::vector<std::vector<uint32_t>>&& test_tokens,
    bool use_sparse_inference) {
  SingleUnitDatasetContext single_predict_context(std::move(test_data),
                                                  std::move(test_tokens));

  verifyCanPredict(single_predict_context, /* has_labels = */ false,
                   /* returning_activations = */ true,
                   /* num_metrics_tracked = */ 0);

  prepareToProcessBatches(/* batch_size = */ 1, use_sparse_inference);

  // TODO(josh/Nick): This try catch is kind of a hack, we should really use
  // some sort of RAII training context object whose destructor will
  // automatically delete the training state
  try {
    single_predict_context.setInputs(/* batch_idx = */ 0, _inputs,
                                     _token_inputs);
    forward(/* vec_index = */ 0, nullptr);
    BoltVector output_copy = _output->getOutputVector(
        /* vec_index = */ 0);
    cleanupAfterBatchProcessing();
    return output_copy;
  } catch (const std::exception& e) {
    cleanupAfterBatchProcessing();
    throw;
  }
}

void BoltGraph::processInferenceBatch(uint64_t batch_size,
                                      const BoltBatch* batch_labels,
                                      MetricAggregator& metrics) {
  // Either we shouldn't track any metrics or there need to be labels
  assert((metrics.getNumMetricsTracked() == 0) || (batch_labels != nullptr));

#pragma omp parallel for default(none) shared(batch_size, batch_labels, metrics)
  for (uint64_t vec_id = 0; vec_id < batch_size; vec_id++) {
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

void BoltGraph::processOutputCallback(
    const std::optional<std::function<void(const BoltVector&)>>&
        output_callback,
    uint32_t batch_size) {
  if (output_callback) {
    for (uint32_t vec_id_in_batch = 0; vec_id_in_batch < batch_size;
         vec_id_in_batch++) {
      const auto& current_output_vec =
          _output->getOutputVector(vec_id_in_batch);
      output_callback.value()(current_output_vec);
    }
  }
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

void BoltGraph::verifyCanTrain(const DatasetContext& train_context) {
  if (!graphCompiled()) {
    throw std::logic_error("Graph must be compiled before training");
  }

  if (!train_context.labels()) {
    throw std::invalid_argument("Must pass in labels for training.");
  }

  verifyInputForGraph(train_context);
}

void BoltGraph::verifyCanGetInputGradientSingle(
    const DatasetContextBase& single_input_gradients_context, bool best_index,
    uint32_t num_output_nonzeros) {
  if (!graphCompiled()) {
    throw std::logic_error(
        "Graph must be compiled before getting input gradients");
  }
  if (!best_index && num_output_nonzeros < 2) {
    throw std::invalid_argument(
        "The sparse output dimension should be atleast 2 to call "
        "getSecondHighestActivationId.");
  }
  verifyInputForGraph(single_input_gradients_context);
}

void BoltGraph::verifyCanGetInputGradients(
    const DatasetContext& input_gradients_context,
    uint32_t required_labels_size, uint32_t input_data_len, bool best_index,
    uint32_t num_output_nonzeros) {
  if (!graphCompiled()) {
    throw std::logic_error(
        "Graph must be compiled before getting input gradients");
  }
  if ((required_labels_size != 0) && (required_labels_size != input_data_len)) {
    throw std::invalid_argument("Length of required_labels " +
                                std::to_string(required_labels_size) +
                                "does not match length of provided dataset." +
                                std::to_string(input_data_len));
  }
  if (!best_index && num_output_nonzeros < 2) {
    throw std::invalid_argument(
        "The sparse output dimension should be atleast 2 to call "
        "getSecondHighestActivationId.");
  }
  verifyInputForGraph(input_gradients_context);
}

void BoltGraph::verifyCanPredict(const DatasetContextBase& predict_context,
                                 bool has_labels, bool returning_activations,
                                 uint32_t num_metrics_tracked) {
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

  verifyInputForGraph(predict_context);
}

void BoltGraph::verifyInputForGraph(const DatasetContextBase& context) {
  if (context.numVectorDatasets() != _inputs.size()) {
    throw std::invalid_argument(
        "Wrong number of dataset inputs, expected " +
        std::to_string(_inputs.size()) + " but received " +
        std::to_string(context.numVectorDatasets()) + ".");
  }

  if (context.numTokenDatasets() != _token_inputs.size()) {
    throw std::invalid_argument("Wrong number of token inputs, expected " +
                                std::to_string(_token_inputs.size()) +
                                " but received " +
                                std::to_string(_inputs.size()) + ".");
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

template void BoltGraph::serialize(cereal::BinaryInputArchive&);
template void BoltGraph::serialize(cereal::BinaryOutputArchive&);

template <class Archive>
void BoltGraph::serialize(Archive& archive) {
  archive(_nodes, _output, _inputs, _token_inputs,
          _internal_fully_connected_layers, _loss, _epoch_count, _batch_cnt);
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