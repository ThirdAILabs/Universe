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
constexpr uint32_t RehashAutoTuneThreshold = 100000;
constexpr uint32_t RehashAutoTuneFactor1 = 100;
constexpr uint32_t RehashAutoTuneFactor2 = 20;

FullyConnectedNetwork::FullyConnectedNetwork(
    std::vector<FullyConnectedLayerConfig> configs, uint32_t input_dim)
    : _configs(std::move(configs)),
      _input_dim(input_dim),
      _iter(0),
      _epoch_count(0) {
  auto start = std::chrono::high_resolution_clock::now();

  _sparse_inference = false;
  _num_layers = _configs.size();
  _layers = new FullyConnectedLayer*[_num_layers];
  _states =
      new BoltBatch[_num_layers - 1]();  // No stored state for output layer

  std::cout << "====== Building Fully Connected Network ======" << std::endl;

  for (uint32_t i = 0; i < _num_layers; i++) {
    uint64_t prev_dim = (i > 0) ? _configs[i - 1].dim : _input_dim;
    if (i < _num_layers - 1) {
      if (_configs[i].act_func == ActivationFunc::Softmax) {
        throw std::invalid_argument(
            "Softmax activation function is not supported for hidden layers.");
      }
    }

    std::cout << _configs[i] << std::endl;
    _layers[i] = new FullyConnectedLayer(_configs[i], prev_dim);
  }

  auto end = std::chrono::high_resolution_clock::now();

  std::cout
      << "Initialized Network in "
      << std::chrono::duration_cast<std::chrono::seconds>(end - start).count()
      << " seconds" << std::endl;
  std::cout << "==============================" << std::endl;
}

std::vector<int64_t> FullyConnectedNetwork::train(
    const dataset::InMemoryDataset<dataset::BoltInputBatch>& train_data,
    float learning_rate, uint32_t epochs, uint32_t rehash_in,
    uint32_t rebuild_in) {
  uint32_t rehash = rehash_in;
  if (rehash_in == 0) {
    if (train_data.len() < RehashAutoTuneThreshold) {
      rehash = train_data.len() / RehashAutoTuneFactor2;
    } else {
      rehash = train_data.len() / RehashAutoTuneFactor1;
    }
  }
  uint32_t rebuild = rebuild_in != 0 ? rebuild_in : (train_data.len() / 4);

  uint32_t batch_size = train_data[0].getBatchSize();

  std::cout << "\nTraining Parameters: batch_size=" << batch_size
            << ", learning_rate=" << learning_rate << ", epochs=" << epochs
            << ", rehash=" << rehash << ", rebuild=" << rebuild << std::endl;

  // Take max with 1 so that we don't get 0 causing a floating point error.
  uint32_t rebuild_batch = std::max<uint32_t>(rebuild / batch_size, 1);
  uint32_t rehash_batch = std::max<uint32_t>(rehash / batch_size, 1);

  uint64_t num_train_batches = train_data.numBatches();

  // Because of how the datasets are read we know that all batches will not have
  // a batch size larger than this so we can just set the batch size here.
  this->createBatchStates(batch_size, false);

  std::vector<int64_t> time_per_epoch;

  BoltBatch outputs = _layers[_num_layers - 1]->createBatchState(batch_size);

  CategoricalCrossEntropyLoss loss;

  for (uint32_t epoch = 0; epoch < epochs; epoch++) {
    std::cout << "\nEpoch " << (_epoch_count + 1) << ':' << std::endl;
    ProgressBar bar(num_train_batches);
    auto train_start = std::chrono::high_resolution_clock::now();

    for (uint32_t batch = 0; batch < num_train_batches; batch++) {
      if (_iter % 1000 == 999) {
        shuffleRandomNeurons();
      }

      dataset::BoltInputBatch& inputs =
          const_cast<dataset::BoltInputBatch&>(train_data[batch]);

#pragma omp parallel for default(none) shared(inputs, outputs, loss)
      for (uint32_t i = 0; i < inputs.getBatchSize(); i++) {
        this->forward(i, inputs[i], outputs[i], &inputs.labels(i));

        loss.loss(outputs[i], inputs.labels(i), inputs.getBatchSize());

        this->backpropagate<true>(i, inputs[i], outputs[i]);
      }

      this->updateParameters(learning_rate);

      if (!_sparse_inference) {
        if (_iter % rebuild_batch == (rebuild_batch - 1)) {
          reBuildHashFunctions();
          buildHashTables();
        } else if (_iter % rehash_batch == (rehash_batch - 1)) {
          buildHashTables();
        }
      }

      bar.increment();
    }

    auto train_end = std::chrono::high_resolution_clock::now();
    int64_t epoch_time = std::chrono::duration_cast<std::chrono::seconds>(
                             train_end - train_start)
                             .count();

    time_per_epoch.push_back(epoch_time);
    std::cout << std::endl
              << "Processed " << num_train_batches << " training batches in "
              << epoch_time << " seconds" << std::endl;
    _epoch_count++;
  }
  return time_per_epoch;
}

float FullyConnectedNetwork::predict(
    const dataset::InMemoryDataset<dataset::BoltInputBatch>& test_data,
    uint32_t batch_limit) {
  uint32_t batch_size = test_data[0].getBatchSize();

  uint64_t num_test_batches = std::min(test_data.numBatches(), batch_limit);

  // Because of how the datasets are read we know that all batches will not have
  // a batch size larger than this so we can just set the batch size here.
  this->createBatchStates(batch_size, true);

  BoltBatch outputs = _layers[_num_layers - 1]->createBatchState(
      batch_size, !_layers[_num_layers - 1]->isForceSparsity());

  std::atomic<uint32_t> correct{0};
  ProgressBar bar(num_test_batches);

  auto test_start = std::chrono::high_resolution_clock::now();
  for (uint32_t batch = 0; batch < num_test_batches; batch++) {
    dataset::BoltInputBatch& inputs =
        const_cast<dataset::BoltInputBatch&>(test_data[batch]);

#pragma omp parallel for default(none) shared(inputs, outputs, correct)
    for (uint32_t i = 0; i < inputs.getBatchSize(); i++) {
      this->forward(i, inputs[i], outputs[i], nullptr);

      const float* activations = outputs[i].activations;
      float max_act = std::numeric_limits<float>::min();
      uint32_t pred = 0;
      for (uint32_t k = 0; k < outputs[i].len; k++) {
        if (activations[k] > max_act) {
          max_act = activations[k];
          // Since sparsity is set to 1.0, the layer is dense and we can use i
          // instead of indices[i]
          if (_layers[_num_layers - 1]->isForceSparsity()) {
            pred = outputs[i].active_neurons[k];
          } else {
            pred = k;
          }
        }
      }

      const uint32_t* label_start = inputs.labels(i).active_neurons;
      const uint32_t* label_end =
          inputs.labels(i).active_neurons + inputs.labels(i).len;
      if (std::find(label_start, label_end, pred) != label_end) {
        correct++;
      }
    }

    bar.increment();
  }

  auto test_end = std::chrono::high_resolution_clock::now();
  // Anshu: Inference times in milliseconds
  int64_t test_time = std::chrono::duration_cast<std::chrono::milliseconds>(
                          test_end - test_start)
                          .count();
  std::cout << std::endl
            << "Processed " << num_test_batches << " test batches in "
            << test_time << " milliseconds" << std::endl;

  uint32_t num_vecs = std::min(num_test_batches * batch_size, test_data.len());
  float accuracy = static_cast<float>(correct) / num_vecs;
  std::cout << "Accuracy: " << accuracy << " (" << correct << "/" << num_vecs
            << ")" << std::endl;

  return accuracy;
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

void FullyConnectedNetwork::createBatchStates(uint32_t batch_size,
                                              bool force_dense) {
  for (uint32_t l = 0; l < _num_layers - 1; l++) {
    _states[l] = _layers[l]->createBatchState(batch_size, force_dense);
  }
}

void FullyConnectedNetwork::updateParameters(float learning_rate) {
  ++_iter;
  for (uint32_t layer = 0; layer < _num_layers; layer++) {
    _layers[layer]->updateParameters(learning_rate, _iter, BETA1, BETA2, EPS);
  }
}

void FullyConnectedNetwork::reBuildHashFunctions() {
  for (uint32_t l = 0; l < _num_layers; l++) {
    _layers[l]->reBuildHashFunction();
  }
}

void FullyConnectedNetwork::buildHashTables() {
  for (uint32_t l = 0; l < _num_layers; l++) {
    _layers[l]->buildHashTables();
  }
}

void FullyConnectedNetwork::shuffleRandomNeurons() {
  for (uint32_t i = 0; i < _num_layers; i++) {
    _layers[i]->shuffleRandNeurons();
  }
}

FullyConnectedNetwork::~FullyConnectedNetwork() {
  for (uint32_t i = 0; i < _num_layers; i++) {
    delete _layers[i];
  }

  delete[] _states;

  delete[] _layers;
}

}  // namespace thirdai::bolt
