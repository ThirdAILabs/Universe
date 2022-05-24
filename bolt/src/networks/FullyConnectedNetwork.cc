#include "FullyConnectedNetwork.h"
#include <bolt/src/layers/BoltVector.h>
#include <bolt/src/layers/ConvLayer.h>
#include <bolt/src/layers/FullyConnectedLayer.h>
#include <bolt/src/loss_functions/LossFunctions.h>
#include <bolt/src/utils/ProgressBar.h>
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <stdexcept>

namespace thirdai::bolt {

FullyConnectedNetwork::FullyConnectedNetwork(SequentialConfigList configs,
                                             uint32_t input_dim)
    : _input_dim(input_dim),
      _num_layers(configs.size()),
      _sparse_inference_enabled(false) {
  auto start = std::chrono::high_resolution_clock::now();

  std::cout << "====== Building Fully Connected Network ======" << std::endl;

  for (uint32_t i = 0; i < _num_layers; i++) {
    std::cout << *configs[i] << std::endl;

    uint64_t prev_dim = i == 0 ? input_dim : configs[i - 1]->getDim();

    // if FullyConnectedConfig
    if (auto fully_connected_config =
            std::dynamic_pointer_cast<FullyConnectedLayerConfig>(configs[i])) {
      _layers.push_back(std::make_shared<FullyConnectedLayer>(
          *fully_connected_config, prev_dim));
      // if ConvConfig
    } else if (auto conv_config =
                   std::static_pointer_cast<ConvLayerConfig>(configs[i])) {
      if (i == _num_layers - 1) {
        throw std::invalid_argument("ConvLayer not supported as final layer.");
      }
      uint64_t prev_channels;
      uint64_t prev_sparse_channels;
      if (i == 0) {
        // TODO(david): handle 3d input, change hardcoded input channels
        prev_channels = 3;
        prev_sparse_channels = 3;
      } else {
        if (auto prev_conv_config =
                std::dynamic_pointer_cast<ConvLayerConfig>(configs[i - 1])) {
          prev_channels = prev_conv_config->num_filters;
          prev_sparse_channels =
              prev_conv_config->num_filters * prev_conv_config->sparsity;
        } else {
          throw std::invalid_argument(
              "ConvLayer can only come after input or another ConvLayer");
        }
      }
      auto next_conv_config =
          std::dynamic_pointer_cast<ConvLayerConfig>(configs[i + 1]);
      std::pair<uint32_t, uint32_t> next_kernel_size =
          next_conv_config ? next_conv_config->kernel_size
                           : std::pair<uint32_t, uint32_t>(1, 1);

      _layers.push_back(
          std::make_shared<ConvLayer>(*conv_config, prev_dim, prev_channels,
                                      prev_sparse_channels, next_kernel_size));
    }
  }

  auto end = std::chrono::high_resolution_clock::now();

  std::cout
      << "Initialized Network in "
      << std::chrono::duration_cast<std::chrono::seconds>(end - start).count()
      << " seconds" << std::endl;
  std::cout << "==============================" << std::endl;
}

void FullyConnectedNetwork::trainOnStreamingDataset(
    dataset::StreamingDataset& dataset, const LossFunction& loss_fn,
    float learning_rate) {
  uint32_t batch_size = dataset.getMaxBatchSize();
  initializeNetworkState(batch_size, false);

  // TODO(nicholas): set these to something
  uint32_t rehash_batch = 0, rebuild_batch = 0;

  BoltBatch outputs = getOutputs(batch_size, false);

  while (auto batch = dataset.nextBatch()) {
    if (_batch_iter % 1000 == 999) {
      shuffleRandomNeurons();
    }

    bolt::BoltBatch& batch_inputs = batch->first;

    const BoltBatch& batch_labels = batch->second;

#pragma omp parallel for default(none) \
    shared(batch_inputs, batch_labels, outputs, loss_fn)
    for (uint32_t vec_id = 0; vec_id < batch_inputs.getBatchSize(); vec_id++) {
      forward(vec_id, batch_inputs, outputs[vec_id], &batch_labels[vec_id]);

      loss_fn.lossGradients(outputs[vec_id], batch_labels[vec_id],
                            batch_inputs.getBatchSize());

      backpropagate(vec_id, batch_inputs, outputs[vec_id]);
    }

    updateParameters(learning_rate, ++_batch_iter);

    if (_batch_iter % rebuild_batch == (rebuild_batch - 1)) {
      reBuildHashFunctions();
      buildHashTables();
    } else if (_batch_iter % rehash_batch == (rehash_batch - 1)) {
      buildHashTables();
    }
  }
}

std::vector<uint32_t> FullyConnectedNetwork::testOnStreamingDataset(
    dataset::StreamingDataset& dataset) {
  uint32_t batch_size = dataset.getMaxBatchSize();
  initializeNetworkState(batch_size, true);

  BoltBatch outputs = getOutputs(batch_size, true);

  std::vector<uint32_t> predictions;

  while (auto batch = dataset.nextBatch()) {
    const BoltBatch& inputs = batch->first;
    std::vector<uint32_t> batch_predictions(inputs.getBatchSize());

#pragma omp parallel for default(none) \
    shared(inputs, outputs, batch_predictions)
    for (uint32_t vec_id = 0; vec_id < inputs.getBatchSize(); vec_id++) {
      // We set labels to nullptr so that they are not used in sampling during
      // inference.
      forward(vec_id, inputs, outputs[vec_id], /*labels=*/nullptr);

      uint32_t pred = 0;
      float max_act = 0.0;
      for (uint32_t i = 0; i < outputs[vec_id].len; i++) {
        if (outputs[vec_id].activations[i] > max_act) {
          pred = outputs[vec_id].active_neurons[i];
          max_act = outputs[vec_id].activations[i];
        }
      }
      batch_predictions[vec_id] = pred;
    }

    predictions.insert(predictions.end(), batch_predictions.begin(),
                       batch_predictions.end());
  }

  return predictions;
}

void FullyConnectedNetwork::forward(uint32_t batch_index,
                                    const BoltVector& input, BoltVector& output,
                                    const BoltVector* labels) {
  for (uint32_t i = 0; i < _num_layers; i++) {
    if (i == 0 && _num_layers == 1) {  // First and last layer
      _layers[0]->forward(input, output, labels);
    } else if (i == 0) {  // First layer
      _layers[0]->forward(input, _states[0][batch_index], nullptr);
    } else if (i == _num_layers - 1) {  // Last layer
      _layers[i]->forward(_states[i - 1][batch_index], output, labels);
    } else {  // Middle layer
      _layers[i]->forward(_states[i - 1][batch_index], _states[i][batch_index],
                          nullptr);
    }
  }
}

template void FullyConnectedNetwork::backpropagate<true>(uint32_t, BoltVector&,
                                                         BoltVector&);
template void FullyConnectedNetwork::backpropagate<false>(uint32_t, BoltVector&,
                                                          BoltVector&);

template <bool FROM_INPUT>
void FullyConnectedNetwork::backpropagate(uint32_t batch_index,
                                          BoltVector& input,
                                          BoltVector& output) {
  for (uint32_t i = _num_layers; i > 0; i--) {
    uint32_t layer = i - 1;
    if (layer == 0 && _num_layers == 1) {  // First and last layer
      if (FROM_INPUT) {
        _layers[0]->backpropagateInputLayer(input, output);
      } else {
        _layers[0]->backpropagate(input, output);
      }
    } else if (layer == 0) {  // First layer
      if (FROM_INPUT) {
        _layers[0]->backpropagateInputLayer(input, _states[0][batch_index]);
      } else {
        _layers[0]->backpropagate(input, _states[0][batch_index]);
      }
    } else if (layer == _num_layers - 1) {  // Last layer
      _layers[layer]->backpropagate(_states[layer - 1][batch_index], output);
    } else {  // Middle layer
      _layers[layer]->backpropagate(_states[layer - 1][batch_index],
                                    _states[layer][batch_index]);
    }
  }
}

void FullyConnectedNetwork::initializeNetworkState(uint32_t batch_size,
                                                   bool force_dense) {
  _states.clear();
  for (uint32_t l = 0; l < _num_layers - 1; l++) {
    _states.push_back(_layers[l]->createBatchState(
        batch_size, useDenseComputations(force_dense)));
  }
}

}  // namespace thirdai::bolt
