#include "DistributedModel.h"
#include <bolt/src/metrics/Metric.h>
#include <bolt/src/networks/FullyConnectedNetwork.h>
#include <bolt/src/utils/ProgressBar.h>
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
uint32_t DistributedModel::prepareNodeForDistributedTraining(
    std::shared_ptr<dataset::InMemoryDataset<bolt::BoltBatch>>& train_data,
    const dataset::BoltDatasetPtr& train_labels, uint32_t rehash,
    uint32_t rebuild, bool verbose) {
  _train_data = train_data;
  _train_labels = train_labels;
  uint32_t batch_size = _train_data->at(0).getBatchSize();
  _rebuild_batch = getRebuildBatch(rebuild, batch_size, train_data->len());
  _rehash_batch = getRehashBatch(rehash, batch_size, train_data->len());

  initTrainDatastructures();

  // Because of how the datasets are read we know that all batches will not have
  // a batch size larger than this so we can just set the batch size here.
  initializeNetworkState(batch_size, true);
  _outputs = getOutputs(batch_size, true);

  if (verbose) {
    std::cout << "Distributed Network initialization done on this Node"
              << std::endl;
  }
  return train_data->numBatches();
}

void DistributedModel::calculateGradientSingleNode(
    uint32_t batch_idx, const LossFunction& loss_fn) {
  bolt::BoltBatch& batch_inputs = _train_data->at(batch_idx);

  const BoltBatch& batch_labels = _train_labels->at(batch_idx);

#pragma omp parallel for default(none) \
    shared(batch_inputs, batch_labels, loss_fn)
  for (uint32_t vec_id = 0; vec_id < batch_inputs.getBatchSize(); vec_id++) {
    forward(vec_id, batch_inputs, _outputs[vec_id], &batch_labels[vec_id]);

    loss_fn.lossGradients(_outputs[vec_id], batch_labels[vec_id],
                          batch_inputs.getBatchSize());
    backpropagate(vec_id, batch_inputs, _outputs[vec_id]);
  }
}

void DistributedModel::updateParametersSingleNode(float learning_rate) {
  updateParameters(learning_rate, ++_batch_iter);
  updateSampling(_rehash_batch, _rebuild_batch);
}

uint32_t DistributedModel::numLayers() const { return _num_layers; }

float* DistributedModel::getWeights(uint32_t layer_index) {
  return _layers.at(layer_index)->getWeights();
}

float* DistributedModel::getBiases(uint32_t layer_index) {
  return _layers.at(layer_index)->getBiases();
}

float* DistributedModel::getWeightsGradient(uint32_t layer_index) {
  return _layers.at(layer_index)->getWeightsGradient();
}

float* DistributedModel::getBiasesGradient(uint32_t layer_index) {
  return _layers.at(layer_index)->getBiasesGradient();
}

void DistributedModel::setWeights(uint32_t layer_index, const float* data) {
  _layers.at(layer_index)->setWeights(data);
}

void DistributedModel::setBiases(uint32_t layer_index, const float* data) {
  _layers.at(layer_index)->setBiases(data);
}

void DistributedModel::setWeightGradients(uint32_t layer_index,
                                          const float* data) {
  _layers.at(layer_index)->setWeightGradients(data);
}

void DistributedModel::setBiasesGradients(uint32_t layer_index,
                                          const float* data) {
  _layers.at(layer_index)->setBiasesGradients(data);
}

uint32_t DistributedModel::getDim(uint32_t layer_index) const {
  return _layers.at(layer_index)->getDim();
}

uint32_t DistributedModel::getInputDim() const { return _input_dim; }

}  // namespace thirdai::bolt