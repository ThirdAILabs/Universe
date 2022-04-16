#include "FullyConnectedNetwork.h"
#include <bolt/src/layers/ConvLayer.h>
#include <bolt/src/loss_functions/LossFunctions.h>
#include <bolt/src/utils/ProgressBar.h>
#include <algorithm>
#include <atomic>
#include <chrono>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <wrappers/src/LicenseWrapper.h>

namespace thirdai::bolt {

FullyConnectedNetwork::FullyConnectedNetwork(
    std::vector<FullyConnectedLayerConfig> configs, uint32_t input_dim)
    : _input_dim(input_dim),
      _num_layers(configs.size()),
      _sparse_inference_enabled(false) {

  thirdai::licensing::LicenseWrapper::checkLicense();

  auto start = std::chrono::high_resolution_clock::now();

  std::cout << "====== Building Fully Connected Network ======" << std::endl;

  for (uint32_t cur_layer = 0; cur_layer < _num_layers; cur_layer++) {
    bool is_conv_layer = configs[cur_layer].isConvLayer();
    if (cur_layer < _num_layers - 1) {
      if (configs[cur_layer].act_func == ActivationFunction::Softmax) {
        throw std::invalid_argument(
            "Softmax activation function is not supported for hidden layers.");
      }
    } else if (cur_layer == _num_layers - 1 && is_conv_layer) {
      throw std::invalid_argument("ConvLayer not supported as final layer.");
    }

    uint64_t prev_dim;
    std::cout << configs[cur_layer] << std::endl;
    if (is_conv_layer) {
      uint64_t prev_num_filters;
      uint64_t prev_num_sparse_filters;
      if (cur_layer > 0) {
        bool prev_is_conv_layer = configs[cur_layer - 1].isConvLayer();
        if (prev_is_conv_layer) {
          prev_dim =
              configs[cur_layer - 1].dim * configs[cur_layer - 1].num_patches;
          prev_num_filters = configs[cur_layer - 1].dim;
          prev_num_sparse_filters =
              configs[cur_layer - 1].dim * configs[cur_layer - 1].sparsity;
        } else {
          prev_dim = configs[cur_layer - 1].dim;
          // TODO(david) edge case for when convlayer comes after another
          // non-conv layer, what is prev_num_filters??
        }
      } else {
        prev_dim = input_dim;
        prev_num_filters =
            1;  // TODO(david) change input dim to vector to accept 3d input?
                // then prev_num_filters = num_input_channels
        prev_num_sparse_filters = 1;
      }

      bool next_is_conv_layer = configs[cur_layer + 1].isConvLayer();
      std::pair<uint32_t, uint32_t> next_kernel_size =
          next_is_conv_layer ? configs[cur_layer + 1].kernel_size
                             : std::make_pair(static_cast<uint32_t>(1),
                                              static_cast<uint32_t>(1));
      _layers.push_back(std::make_shared<ConvLayer>(
          configs[cur_layer], prev_dim, prev_num_filters,
          prev_num_sparse_filters, next_kernel_size));
    } else {
      if (cur_layer > 0) {
        bool prev_is_conv_layer = configs[cur_layer - 1].isConvLayer();
        if (prev_is_conv_layer) {
          prev_dim =
              configs[cur_layer - 1].dim * configs[cur_layer - 1].num_patches;
        } else {
          prev_dim = configs[cur_layer - 1].dim;
        }
      } else {
        prev_dim = input_dim;
      }
      _layers.push_back(
          std::make_shared<FullyConnectedLayer>(configs[cur_layer], prev_dim));
    }
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
      _layers[0]->forward(input, _states[0][batch_index]);
    } else if (i == _num_layers - 1) {  // Last layer
      _layers[i]->forward(_states[i - 1][batch_index], output, labels);
    } else {  // Middle layer
      _layers[i]->forward(_states[i - 1][batch_index], _states[i][batch_index]);
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
