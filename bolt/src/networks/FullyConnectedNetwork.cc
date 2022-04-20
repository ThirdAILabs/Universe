#include "FullyConnectedNetwork.h"
#include <bolt/src/loss_functions/LossFunctions.h>
#include <bolt/src/utils/ProgressBar.h>
#include <algorithm>
#include <atomic>
#include <chrono>
#include <iostream>
#include <limits>
#include <stdexcept>

namespace thirdai::bolt {

FullyConnectedNetwork::FullyConnectedNetwork(
    std::vector<std::shared_ptr<SequentialLayerConfig>> configs,
    uint32_t input_dim)
    : _input_dim(input_dim),
      _num_layers(configs.size()),
      _sparse_inference_enabled(false) {
  auto start = std::chrono::high_resolution_clock::now();

  std::cout << "====== Building Fully Connected Network ======" << std::endl;

  for (uint32_t i = 0; i < _num_layers; i++) {
    std::cout << *configs[i].get() << std::endl;

    std::shared_ptr<SequentialLayerConfig> prev_config =
        i == 0 ? std::dynamic_pointer_cast<SequentialLayerConfig>(
                     std::make_shared<InputConfig>(input_dim))
               : configs[i - 1];
    std::shared_ptr<SequentialLayerConfig> next_config =
        i != _num_layers - 1 ? configs[i + 1] : nullptr;

    _layers.push_back(configs[i]->createLayer(prev_config, next_config));
  }

  auto end = std::chrono::high_resolution_clock::now();

  std::cout
      << "Initialized Network in "
      << std::chrono::duration_cast<std::chrono::seconds>(end - start).count()
      << " seconds" << std::endl;
  std::cout << "==============================" << std::endl;
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
