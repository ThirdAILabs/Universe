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
#include <optional>
#include <sstream>
#include <stdexcept>
#include <utility>

namespace thirdai::bolt {

FullyConnectedNetwork::FullyConnectedNetwork(SequentialConfigList configs,
                                             uint32_t input_dim,
                                             bool is_distributed)

    : _input_dim(input_dim), _num_layers(configs.size()) {
  auto start = std::chrono::high_resolution_clock::now();

  std::cout << "Initializing Bolt network..." << std::endl;

  for (uint32_t i = 0; i < _num_layers; i++) {
    uint64_t prev_dim = i == 0 ? input_dim : configs[i - 1]->getDim();

    // if FullyConnectedConfig
    if (auto fully_connected_config =
            std::dynamic_pointer_cast<FullyConnectedLayerConfig>(configs[i])) {
      _layers.push_back(std::make_shared<FullyConnectedLayer>(
          *fully_connected_config, prev_dim, is_distributed));
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

std::pair<std::vector<std::vector<float>>,
          std::optional<std::vector<std::vector<uint32_t>>>>
FullyConnectedNetwork::getInputGradients(
    std::shared_ptr<dataset::InMemoryDataset<BoltBatch>>& input_dataset,
    const LossFunction& loss_fn, bool best_index,
    const std::vector<uint32_t>& required_labels) {
  uint64_t num_batches = input_dataset->numBatches();
  if (!required_labels.empty() &&
      (required_labels.size() != input_dataset->len())) {
    throw std::invalid_argument("Length of required_labels " +
                                std::to_string(required_labels.size()) +
                                "does not match length of provided dataset." +
                                std::to_string(input_dataset->len()));
  }
  if (!best_index && getInferenceOutputDim(true) < 2) {
    throw std::invalid_argument(
        "The sparse output dimension should be atleast 2 to call "
        "getSecondBestId.");
  }
  // Because of how the datasets are read we know that all batches will not
  // have a batch size larger than this so we can just set the batch size
  // here.
  initializeNetworkState(input_dataset->at(0).getBatchSize(),
                         /*use_sparsity= */ true);
  std::vector<std::vector<float>> input_dataset_grad;
  std::vector<std::vector<uint32_t>> input_dataset_indices;
  for (uint64_t batch_id = 0; batch_id < num_batches; batch_id++) {
    BoltBatch output = getOutputs(input_dataset->at(batch_id).getBatchSize(),
                                  /*use_sparsity= */ true);
    for (uint32_t vec_id = 0;
         vec_id < input_dataset->at(batch_id).getBatchSize(); vec_id++) {
      std::vector<float> vec_grad(input_dataset->at(batch_id)[vec_id].len, 0.0);
      // Assigning the vec_grad data() to gradients so that we dont have to
      // worry about initializing and then freeing the memory.
      input_dataset->at(batch_id)[vec_id].gradients = vec_grad.data();
      uint32_t required_index;
      /*
      we are taking the second best index to know how change in input vector
      values affects the prediction to flip to second highest activation. And
      required_labels is essential because for some of the cases we know the
      correct output labels, and best index is used to explain the
      prediction.
      */
      /*
      If the required_labels are empty, then we have to find the required_index
      by output activations, for that we need to do forward pass before creating
      the batch_label, but if the required_labels are not empty and for some ,If
      the required label position is not present in the output active neurons ,
      then calculating the gradients with respect to that label doesnot make
      sense, because loss is only calculated with respect to active neurons, to
      ensure that output has active neuron at the position of required label we
      are creating batch_label before forward pass and passing to it, because
      forward pass ensures to have active neurons at the metioned label index.
      */
      BoltVector batch_label;
      if (required_labels.empty()) {
        forward(vec_id, input_dataset->at(batch_id), output[vec_id],
                /*labels = */ nullptr);
        if (best_index) {
          required_index = output[vec_id].getIdWithHighestActivation();
        } else {
          required_index = output[vec_id].getSecondBestId();
        }
        batch_label = BoltVector::makeSparseVector(
            std::vector<uint32_t>{required_index}, std::vector<float>{1.0});
      } else {
        required_index =
            required_labels[batch_id * input_dataset->at(0).getBatchSize() +
                            vec_id];
        if (required_index >= getOutputDim()) {
          throw std::invalid_argument(
              "Cannot pass required index " + std::to_string(required_index) +
              " to getInputGradients for network with output dim " +
              std::to_string(getOutputDim()));
        }
        batch_label = BoltVector::makeSparseVector(
            std::vector<uint32_t>{required_index}, std::vector<float>{1.0});
        forward(vec_id, input_dataset->at(batch_id), output[vec_id],
                /*labels = */ &batch_label);
      }
      if (!input_dataset->at(batch_id)[vec_id].isDense()) {
        std::vector<uint32_t> vec_indices(
            input_dataset->at(batch_id)[vec_id].active_neurons,
            input_dataset->at(batch_id)[vec_id].active_neurons +
                input_dataset->at(batch_id)[vec_id].len);
        input_dataset_indices.push_back(vec_indices);
      }

      loss_fn.lossGradients(output[vec_id], batch_label,
                            input_dataset->at(batch_id).getBatchSize());

      backpropagateInputForGradients(vec_id, input_dataset->at(batch_id),
                                     output[vec_id]);

      // We reset the gradients to nullptr here to prevent the bolt vector from
      // freeing the memory which is owned by the std::vector we used to store
      // the gradients
      input_dataset->at(batch_id)[vec_id].gradients = nullptr;
      input_dataset_grad.push_back(vec_grad);
    }
  }
  if (input_dataset_indices.empty()) {
    return std::make_pair(input_dataset_grad, std::nullopt);
  }
  return std::make_pair(input_dataset_grad, input_dataset_indices);
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
                                                   bool use_sparsity) {
  _states.clear();
  for (uint32_t l = 0; l < _num_layers - 1; l++) {
    _states.push_back(_layers[l]->createBatchState(
        batch_size, /* use_sparsity= */ use_sparsity));
  }
}

}  // namespace thirdai::bolt