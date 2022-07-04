#include "DistributedModel.h"
#include <bolt/src/metrics/Metric.h>
#include <bolt/src/networks/FullyConnectedNetwork.h>
#include <bolt/src/utils/ProgressBar.h>
#include <dataset/src/batch_types/ClickThroughBatch.h>
#include <algorithm>
#include <cctype>
#include <chrono>
#include <cstdint>
#include <iostream>
#include <stdexcept>

namespace thirdai::bolt {





uint32_t DistributedModel::initTrainDistributed(
    std::shared_ptr<dataset::InMemoryDataset<bolt::BoltBatch>>& train_data,
    const dataset::BoltDatasetPtr& train_labels,
    // Clang tidy is disabled for this line because it wants to pass by
    // reference, but shared_ptrs should not be passed by reference
    uint32_t rehash, uint32_t rebuild, bool verbose) {
  _train_data = train_data;
  _train_labels = train_labels;
  uint32_t batch_size = _train_data->at(0).getBatchSize();
  _rebuild_batch =
      getRebuildBatchDistributed(rebuild, batch_size, train_data->len());
  _rehash_batch =
      getRehashBatchDistributed(rehash, batch_size, train_data->len());

  // Because of how the datasets are read we know that all batches will not have
  // a batch size larger than this so we can just set the batch size here.
  initializeNetworkState(batch_size, false);
  _outputs = getOutputs(batch_size, false);

  if (verbose) {
    std::cout << "Distributed Network initialization done on this Node"
              << std::endl;
  }
  return train_data->numBatches();
}

void DistributedModel::calculateGradientDistributed(
    uint32_t batch, const LossFunction& loss_fn) {
  bolt::BoltBatch& batch_inputs = _train_data->at(batch);

  const BoltBatch& batch_labels = _train_labels->at(batch);


#pragma omp parallel for default(none) \
    shared(batch_inputs, batch_labels, _outputs, loss_fn)
  for (uint32_t vec_id = 0; vec_id < batch_inputs.getBatchSize(); vec_id++) {
    forward(vec_id, batch_inputs, _outputs[vec_id], &batch_labels[vec_id]);

    loss_fn.lossGradients(_outputs[vec_id], batch_labels[vec_id],
                          batch_inputs.getBatchSize());
    backpropagate(vec_id, batch_inputs, _outputs[vec_id]);
  }
}

void DistributedModel::updateParametersDistributed(
    float learning_rate) {
  updateParameters(learning_rate, ++_batch_iter);
  if (_batch_iter % _rebuild_batch == (_rebuild_batch - 1)) {
    reBuildHashFunctions();
    buildHashTables();
  } else if (_batch_iter % _rehash_batch == (_rehash_batch - 1)) {
    buildHashTables();
  }
}

InferenceMetricData DistributedModel::predictDistributed(
    const std::shared_ptr<dataset::InMemoryDataset<bolt::BoltBatch>>& test_data,
    const dataset::BoltDatasetPtr& labels, uint32_t* output_active_neurons,
    float* output_activations, bool use_sparse_inference,
    const std::vector<std::string>& metric_names, bool verbose,
    uint32_t batch_limit) {
  
  assert(output_activations != nullptr || output_active_neurons == nullptr);
  bool compute_metrics = labels != nullptr;

  uint32_t batch_size = test_data->at(0).getBatchSize();

  uint64_t num_test_batches = std::min(test_data->numBatches(), batch_limit);

  uint64_t inference_output_dim = DistributedNetwork.getInferenceOutputDim(use_sparse_inference);

  MetricAggregator metrics(metric_names, verbose);

  // Because of how the datasets are read we know that all batches will not have
  // a batch size larger than this so we can just set the batch size here.
  DistributedNetwork.initializeNetworkState(batch_size, /* use_sparsity= */ use_sparse_inference);
  BoltBatch outputs =
      DistributedNetwork.getOutputs(batch_size, /* use_sparsity= */ use_sparse_inference);

  ProgressBar bar(num_test_batches, verbose);

  auto test_start = std::chrono::high_resolution_clock::now();
  for (uint32_t batch = 0; batch < num_test_batches; batch++) {
    const bolt::BoltBatch& inputs = test_data->at(batch);

    const BoltBatch* batch_labels =
        compute_metrics ? &(*labels)[batch] : nullptr;

    uint32_t* batch_active_neurons =
        output_active_neurons == nullptr
            ? nullptr
            : output_active_neurons + batch * batch_size * inference_output_dim;

    float* batch_activations =
        output_activations == nullptr
            ? nullptr
            : output_activations + batch * batch_size * inference_output_dim;

    processTestBatch(inputs, outputs, batch_labels, batch_active_neurons,
                     batch_activations, inference_output_dim, metrics,
                     compute_metrics);

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

 
inline void DistributedModel::processTestBatch(
    const bolt::BoltBatch& batch_inputs, BoltBatch& outputs,
    const BoltBatch* batch_labels, uint32_t* output_active_neurons,
    float* output_activations, uint64_t inference_output_dim,
    MetricAggregator& metrics, bool compute_metrics) {
#pragma omp parallel for default(none)                                 \
    shared(batch_inputs, batch_labels, outputs, output_active_neurons, \
           output_activations, inference_output_dim, metrics, compute_metrics)
  for (uint32_t vec_id = 0; vec_id < batch_inputs.getBatchSize(); vec_id++) {
    // We set labels to nullptr so that they are not used in sampling during
    // inference.
    DistributedNetwork.forward(vec_id, batch_inputs, outputs[vec_id], /*labels=*/nullptr);

    if (compute_metrics) {
      metrics.processSample(outputs[vec_id], (*batch_labels)[vec_id]);
    }

    if (output_activations != nullptr) {
      assert(outputs[vec_id].len == inference_output_dim);
      const float* start = outputs[vec_id].activations;
      uint32_t offset = vec_id * inference_output_dim;
      std::copy(start, start + outputs[vec_id].len,
                output_activations + offset);
      if (!outputs[vec_id].isDense()) {
        assert(output_active_neurons != nullptr);
        const uint32_t* start = outputs[vec_id].active_neurons;
        std::copy(start, start + outputs[vec_id].len,
                  output_active_neurons + offset);
      }
    }
  }
}

uint32_t DistributedModel::getInferenceOutputDim(bool use_sparse_inference) const {
  return DistributedNetwork.getInferenceOutputDim(use_sparse_inference);
}

uint32_t DistributedModel::getOutputDim() const{
  return DistributedNetwork.getOutputDim();
}

uint32_t DistributedModel::numLayers() const{
    return DistributedNetwork._num_layers;
}

float* DistributedModel::getLayerData(uint32_t layer_index, get_type type){
  
    switch(type){
      case get_weights:
        return DistributedNetwork._layers[layer_index]->getWeights();
      break;
      case get_biases:
        return DistributedNetwork._layers[layer_index]->getBiases();
      break;
      case get_weights_gradients:
        return DistributedNetwork._layers[layer_index]->getWeightsGradient();
      break;
      case get_biases_gradients:
        return DistributedNetwork._layers[layer_index]->getBiasesGradient();
      break;
    }
    throw std::invalid_argument("Wrong argument for getLayerData Function");
  return NULL;
}

void DistributedModel::setLayerData(uint32_t layer_index, const float* set_data, set_type type){
  
    switch(type){
      case set_weights:
        DistributedNetwork._layers[layer_index]->setWeights(set_data);
      break;
      case set_biases:
        DistributedNetwork._layers[layer_index]->setBiases(set_data);
      break;
      case set_weights_gradients:
        DistributedNetwork._layers[layer_index]->setWeightGradients(set_data);
      break;
      case set_biases_gradients:
        DistributedNetwork._layers[layer_index]->setBiasesGradients(set_data);
      break;
    }
}

uint32_t DistributedModel::getDim(uint32_t layer_index) const{
    return DistributedNetwork._layers.at(layer_index)->getDim();
}

uint32_t DistributedModel::getInputDim() const{
  return DistributedNetwork._input_dim;
}

static constexpr uint32_t RehashAutoTuneThreshold = 100000;
static constexpr uint32_t RehashAutoTuneFactor1 = 100;
static constexpr uint32_t RehashAutoTuneFactor2 = 20;

uint32_t DistributedModel::getRehashBatchDistributed(
    uint32_t rehash, uint32_t batch_size, uint32_t data_len) {
  if (rehash == 0) {
    if (data_len < RehashAutoTuneThreshold) {
      rehash = data_len / RehashAutoTuneFactor2;
    } else {
      rehash = data_len / RehashAutoTuneFactor1;
    }
  }
  return std::max<uint32_t>(rehash / batch_size, 1);
}

uint32_t DistributedModel::getRebuildBatchDistributed(
    uint32_t rebuild, uint32_t batch_size, uint32_t data_len) {
  rebuild = rebuild != 0 ? rebuild : (data_len / 4);
  return std::max<uint32_t>(rebuild / batch_size, 1);
}



// The following functions are added to make Bolt Distributed work.
// These functions are going to be extended to python with the help
// of python bindings.

}  // namespace thirdai::bolt