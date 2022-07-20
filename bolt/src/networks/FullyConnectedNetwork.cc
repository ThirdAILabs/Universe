#include "FullyConnectedNetwork.h"
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
#include <sstream>
#include <stdexcept>

namespace thirdai::bolt {

FullyConnectedNetwork::FullyConnectedNetwork(SequentialConfigList configs,
                                             uint32_t input_dim)
    : _input_dim(input_dim), _num_layers(configs.size()) {
  auto start = std::chrono::high_resolution_clock::now();

  std::cout << "Initializing Bolt network..." << std::endl;

  for (uint32_t i = 0; i < _num_layers; i++) {
    uint64_t prev_dim = i == 0 ? input_dim : configs[i - 1]->getDim();

    // if FullyConnectedConfig
    if (auto fully_connected_config =
            std::dynamic_pointer_cast<FullyConnectedLayerConfig>(configs[i])) {
      _layers.push_back(std::make_shared<FullyConnectedLayer>(
          *fully_connected_config, prev_dim));
      // if ConvConfig
    } else if (auto conv_config =
                   std::dynamic_pointer_cast<ConvLayerConfig>(configs[i])) {
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

  this->printSummary();

  std::cout
      << "Initialized Network in "
      << std::chrono::duration_cast<std::chrono::seconds>(end - start).count()
      << " seconds" << std::endl;
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

inline uint32_t getSecondBestIndex(const float* activations, uint32_t dim) {
  float first = std::numeric_limits<float>::min(),
        second = std::numeric_limits<float>::min();
  uint32_t max_id = 0, second_max_id = 0;
  if (dim < 2) {
    throw std::invalid_argument("The output dimension should be atleast 2.");
  }
  for (uint32_t i = 0; i < dim; i++) {
    if (activations[i] > first) {
      second = first;
      second_max_id = max_id;
      first = activations[i];
      max_id = i;
    } else if (activations[i] > second && activations[i] != first) {
      second = activations[i];
      second_max_id = i;
    }
  }
  return second_max_id;
}

void FullyConnectedNetwork::getInputGradientsForBatch(
    BoltBatch& batch_input, BoltBatch& output, const LossFunction& loss_fn,
    uint32_t batch_id, const std::vector<uint32_t>& required_labels,
    std::vector<std::vector<float>>& concatenated_grad, bool want_ratios,
    std::vector<std::vector<float>>& ratios) {
  for (uint32_t vec_id = 0; vec_id < batch_input.getBatchSize(); vec_id++) {
    std::vector<float> vec_grad, ratio_grad;
    // Initializing the input gradients because they were not initialized
    // before. and assigning them to zero because new method gets some random
    // garbage value and gradient calculation uses += operator.
    batch_input[vec_id].gradients = new float[batch_input[vec_id].len];
    for (uint32_t i = 0; i < batch_input[vec_id].len; i++) {
      batch_input[vec_id].gradients[i] = 0;
    }
    forward(vec_id, batch_input, output[vec_id], nullptr);
    uint32_t required_index;
    // we are taking the second best index to know which input features are
    // important by observing input gradients, by flipping the predicted label
    // as second best index.
    if (required_labels.empty()) {
      required_index =
          getSecondBestIndex(output[vec_id].activations, getOutputDim());
    } else {
      required_index =
          (required_labels[batch_id * batch_input.getBatchSize() + vec_id] <=
           getOutputDim() - 1)
              ? required_labels[batch_id * batch_input.getBatchSize() + vec_id]
              : throw std::invalid_argument(
                    "one of the label crossing the output dim");
    }
    BoltVector batch_label = BoltVector::makeSparseVector(
        std::vector<uint32_t>{required_index}, std::vector<float>{1.0});
    loss_fn.lossGradients(output[vec_id], batch_label,
                          batch_input.getBatchSize());
    backpropagateInputForGradients(vec_id, batch_input, output[vec_id]);
    for (uint32_t i = 0; i < batch_input[vec_id].len; i++) {
      vec_grad.push_back(batch_input[vec_id].gradients[i]);
    }
    if (want_ratios) {
      for (uint32_t i = 0; i < batch_input[vec_id].len; i++) {
        ratio_grad.push_back((batch_input[vec_id].gradients[i]) /
                             batch_input[vec_id].activations[i]);
      }
      ratios.push_back(ratio_grad);
    }
    // de allocating the memory and pointing the gradients to nullptr to
    // prevent using invalid memory reference.
    delete[] batch_input[vec_id].gradients;
    batch_input[vec_id].gradients = nullptr;
    concatenated_grad.push_back(vec_grad);
  }
}

std::vector<std::vector<float>> FullyConnectedNetwork::getInputGradients(
    std::shared_ptr<dataset::InMemoryDataset<BoltBatch>>& batch_input,
    const LossFunction& loss_fn, const std::vector<uint32_t>& required_labels) {
  uint64_t num_batches = batch_input->numBatches();
  if (!required_labels.empty() &&
      (required_labels.size() !=
       num_batches * batch_input->at(0).getBatchSize())) {
    throw std::invalid_argument("number of labels does not match");
  }
  // Because of how the datasets are read we know that all batches will not
  // have a batch size larger than this so we can just set the batch size
  // here.
  initializeNetworkState(batch_input->at(0).getBatchSize(), true);
  std::vector<std::vector<float>> concatenated_grad;
  for (uint64_t id = 0; id < num_batches; id++) {
    BoltBatch output = getOutputs(batch_input->at(id).getBatchSize(), true);
    getInputGradientsForBatch(batch_input->at(id), output, loss_fn, id,
                              required_labels, concatenated_grad);
  }
  return concatenated_grad;
}

std::pair<std::vector<std::vector<float>>, std::vector<std::vector<float>>>
FullyConnectedNetwork::getInputGradientsFromStream(
    const std::shared_ptr<dataset::StreamingDataset<BoltBatch>> test_data,
    const LossFunction& loss_fn, uint32_t label_id, bool label_given) {
  uint32_t batch_size = test_data->getMaxBatchSize();
  std::vector<std::vector<float>> concatenated_grad, ratios;
  BoltBatch output = getOutputs(batch_size, true);
  std::vector<uint32_t> temp;
  if (label_given) {
    temp.resize(batch_size, label_id);
  }
  while (auto batch = test_data->nextBatch()) {
    getInputGradientsForBatch(batch->first, output, loss_fn, 0, temp,
                              concatenated_grad, true, ratios);
  }
  return std::make_pair(concatenated_grad, ratios);
}

void FullyConnectedNetwork::initializeNetworkState(uint32_t batch_size,
                                                   bool use_sparsity) {
  _states.clear();
  for (uint32_t l = 0; l < _num_layers - 1; l++) {
    _states.push_back(_layers[l]->createBatchState(
        batch_size, /* use_sparsity= */ use_sparsity));
  }
}

}  // namespace thirdai::bolt