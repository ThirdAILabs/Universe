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

/*
 * This function initializes the network on the node for which it
 * is called. It initializes the network, output array and return
 * number of batches available for training.
 */
uint32_t DistributedModel::initTrainSingleNode(
    std::shared_ptr<dataset::InMemoryDataset<bolt::BoltBatch>>& train_data,
    const dataset::BoltDatasetPtr& train_labels, uint32_t rehash,
    uint32_t rebuild, bool verbose) {
  _train_data = train_data;
  _train_labels = train_labels;
  uint32_t batch_size = _train_data->at(0).getBatchSize();
  _rebuild_batch = DistributedNetwork.getRebuildBatch(rebuild, batch_size,
                                                      train_data->len());
  _rehash_batch =
      DistributedNetwork.getRehashBatch(rehash, batch_size, train_data->len());

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

void DistributedModel::calculateGradientSingleNode(
    uint32_t batch, const LossFunction& loss_fn) {
  bolt::BoltBatch& batch_inputs = _train_data->at(batch);

  const BoltBatch& batch_labels = _train_labels->at(batch);

#pragma omp parallel for default(none) \
    shared(batch_inputs, batch_labels, _outputs, loss_fn)
  for (uint32_t vec_id = 0; vec_id < batch_inputs.getBatchSize(); vec_id++) {
    DistributedNetwork.forward(vec_id, batch_inputs, _outputs[vec_id],
                               &batch_labels[vec_id]);

    loss_fn.lossGradients(_outputs[vec_id], batch_labels[vec_id],
                          batch_inputs.getBatchSize());
    DistributedNetwork.backpropagate(vec_id, batch_inputs, _outputs[vec_id]);
  }
}

void DistributedModel::updateParametersSingleNode(float learning_rate) {
  DistributedNetwork.updateParameters(learning_rate, ++_batch_iter);
  if (_batch_iter % _rebuild_batch == (_rebuild_batch - 1)) {
    reBuildHashFunctions();
    buildHashTables();
  } else if (_batch_iter % _rehash_batch == (_rehash_batch - 1)) {
    buildHashTables();
  }
}

InferenceMetricData DistributedModel::predictSingleNode(
    const std::shared_ptr<dataset::InMemoryDataset<bolt::BoltBatch>>& test_data,
    const dataset::BoltDatasetPtr& labels, uint32_t* output_active_neurons,
    float* output_activations, bool use_sparse_inference,
    const std::vector<std::string>& metric_names, bool verbose,
    uint32_t batch_limit) {
  return DistributedNetwork.predict(test_data, labels, output_active_neurons,
                                    output_activations, use_sparse_inference,
                                    metric_names, verbose, batch_limit);
}

uint32_t DistributedModel::getInferenceOutputDim(
    bool use_sparse_inference) const {
  return DistributedNetwork.getInferenceOutputDim(use_sparse_inference);
}

uint32_t DistributedModel::getOutputDim() const {
  return DistributedNetwork.getOutputDim();
}

uint32_t DistributedModel::numLayers() const {
  return DistributedNetwork._num_layers;
}

float* DistributedModel::getLayerData(uint32_t layer_index, GetType type) {
  switch (type) {
    case GET_WEIGHTS:
      return DistributedNetwork._layers[layer_index]->getWeights();
      break;
    case GET_BIASES:
      return DistributedNetwork._layers[layer_index]->getBiases();
      break;
    case GET_WEIGHT_GRADIENTS:
      return DistributedNetwork._layers[layer_index]->getWeightsGradient();
      break;
    case GET_BIASES_GRADIENTS:
      return DistributedNetwork._layers[layer_index]->getBiasesGradient();
      break;
  }
  throw std::invalid_argument("Wrong argument for getLayerData Function");
  return NULL;
}

void DistributedModel::setLayerData(uint32_t layer_index, const float* set_data,
                                    SetType type) {
  switch (type) {
    case SET_WEIGHTS:
      DistributedNetwork._layers[layer_index]->setWeights(set_data);
      break;
    case SET_BIASES:
      DistributedNetwork._layers[layer_index]->setBiases(set_data);
      break;
    case SET_WEIGHTS_GRADIENTS:
      DistributedNetwork._layers[layer_index]->setWeightGradients(set_data);
      break;
    case SET_BIASES_GRADIENTS:
      DistributedNetwork._layers[layer_index]->setBiasesGradients(set_data);
      break;
  }
}

uint32_t DistributedModel::getDim(uint32_t layer_index) const {
  return DistributedNetwork._layers.at(layer_index)->getDim();
}

uint32_t DistributedModel::getInputDim() const {
  return DistributedNetwork._input_dim;
}

void DistributedModel::setRandomSeed(uint32_t random_seed) const {
  for (uint32_t i = 0; i < DistributedNetwork._num_layers; i++) {
    if (DistributedNetwork._layers[i]->getSparsity() < 1) {
      DistributedNetwork._layers[i]->setSparsity(
          DistributedNetwork._layers[i]->getSparsity(), random_seed);
    }
  }
}

}  // namespace thirdai::bolt