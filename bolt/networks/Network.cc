#include "Network.h"
#include "../utils/ProgressBar.h"
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

Network::Network(std::vector<FullyConnectedLayerConfig> configs,
                 uint64_t input_dim)
    : _configs(std::move(configs)), _input_dim(input_dim), _iter(0) {
  auto start = std::chrono::high_resolution_clock::now();

  _num_layers = _configs.size();
  _layers = new FullyConnectedLayer*[_num_layers];

  std::cout << "====== Building Network ======" << std::endl;

  for (uint32_t i = 0; i < _num_layers; i++) {
    uint64_t prev_dim = (i > 0) ? _configs[i - 1].dim : _input_dim;
    if (i < _num_layers - 1) {
      if (_configs[i].act_func == ActivationFunc::Softmax) {
        throw std::invalid_argument(
            "Softmax activation function is not supported for hidden layers.");
      }
    } else {
      if (_configs[i].act_func == ActivationFunc::ReLU) {
        throw std::invalid_argument(
            "Softmax activation function is required for output layer.");
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

void Network::processTrainingBatch(const Batch& batch, float lr) {
  uint32_t batch_size = batch.batch_size;

#pragma omp parallel for default(none) shared(batch, batch_size)
  for (uint32_t b = 0; b < batch_size; b++) {
    /**
     * 1. Feed Forward
     */
    _layers[0]->feedForward(b, batch.indices[b], batch.values[b], batch.lens[b],
                            nullptr, 0);

    for (uint32_t l = 1; l < _num_layers - 1; l++) {
      FullyConnectedLayer* prev_layer = _layers[l - 1];
      _layers[l]->feedForward(b, prev_layer->getIndices(b),
                              prev_layer->getValues(b), prev_layer->getLen(b),
                              nullptr, 0);
    }
    FullyConnectedLayer* second_to_last_layer = _layers[_num_layers - 2];
    _layers[_num_layers - 1]->feedForward(
        b, second_to_last_layer->getIndices(b),
        second_to_last_layer->getValues(b), second_to_last_layer->getLen(b),
        batch.labels[b], batch.label_lens[b]);

    /**
     * 2. Compute Errors
     */
    _layers[_num_layers - 1]->computeErrors(
        b, batch.batch_size, batch.labels[b], batch.label_lens[b]);

    /**
     * 3. Backpropogation
     */

    for (uint32_t l = _num_layers - 1; l > 0; l--) {
      FullyConnectedLayer* prev_layer = _layers[l - 1];
      _layers[l]->backpropagate(
          b, prev_layer->getIndices(b), prev_layer->getValues(b),
          prev_layer->getErrors(b), prev_layer->getLen(b));
    }

    _layers[0]->backpropagateFirstLayer(b, batch.indices[b], batch.values[b],
                                        batch.lens[b]);
  }

  ++_iter;
  for (uint32_t layer = 0; layer < _num_layers; layer++) {
    _layers[layer]->updateParameters(lr, _iter, BETA1, BETA2, EPS);
  }
}

uint32_t Network::processTestBatch(const Batch& batch) {
  uint32_t batch_size = batch.batch_size;

  for (uint32_t layer = 0; layer < _num_layers; layer++) {
    _layers[layer]->setSparsity(1.0);
  }

  std::atomic<uint32_t> correct{0};

#pragma omp parallel for default(none) shared(batch, batch_size, correct)
  for (uint32_t b = 0; b < batch_size; b++) {
    _layers[0]->feedForward(b, batch.indices[b], batch.values[b], batch.lens[b],
                            nullptr, 0);

    for (uint32_t l = 1; l < _num_layers - 1; l++) {
      FullyConnectedLayer* prev_layer = _layers[l - 1];
      _layers[l]->feedForward(b, prev_layer->getIndices(b),
                              prev_layer->getValues(b), prev_layer->getLen(b),
                              nullptr, 0);
    }
    FullyConnectedLayer* second_to_last_layer = _layers[_num_layers - 2];

    _layers[_num_layers - 1]->feedForward(
        b, second_to_last_layer->getIndices(b),
        second_to_last_layer->getValues(b), second_to_last_layer->getLen(b),
        batch.labels[b], batch.label_lens[b]);

    const float* activations = _layers[_num_layers - 1]->getValues(b);
    float max_act = std::numeric_limits<float>::min();
    uint32_t pred = 0;
    for (uint32_t i = 0; i < _layers[_num_layers - 1]->getLen(b); i++) {
      if (activations[i] > max_act) {
        max_act = activations[i];
        // Since sparsity is set to 1.0, the layer is dense and we can use i
        // instead of indices[i]
        pred = i;
      }
    }

    if (std::find(batch.labels[b], batch.labels[b] + batch.label_lens[b],
                  pred) != batch.labels[b] + batch.label_lens[b]) {
      correct++;
    }
  }

  for (uint32_t layer = 0; layer < _num_layers; layer++) {
    _layers[layer]->setSparsity(_configs[layer].sparsity);
  }

  return correct;
}

void Network::train(uint32_t batch_size, const std::string& train_data,
                    const std::string& test_data, float learning_rate,
                    uint32_t epochs, uint32_t rehash_in, uint32_t rebuild_in,
                    uint32_t max_test_batches) {
  SvmDataset train(train_data, batch_size);
  SvmDataset test(test_data, batch_size);

  uint32_t rehash = rehash_in;
  if (rehash_in == 0) {
    if (train.NumVecs() < RehashAutoTuneThreshold) {
      rehash = train.NumVecs() / RehashAutoTuneFactor2;
    } else {
      rehash = train.NumVecs() / RehashAutoTuneFactor1;
    }
  }
  uint32_t rebuild = rebuild_in != 0 ? rebuild_in : (train.NumVecs() / 4);

  std::cout << "\nTraining Parameters: batch_size=" << batch_size
            << ", learning_rate=" << learning_rate << ", epochs=" << epochs
            << ", rehash=" << rehash << ", rebuild=" << rebuild << std::endl;

  uint64_t intermediate_test_batches =
      std::min<uint64_t>(test.NumBatches(), max_test_batches);
  uint64_t intermediate_test_vecs = std::min<uint64_t>(
      test.NumVecs(), intermediate_test_batches * batch_size);

  // Take max with 1 so that we don't get 0 causing a floating point error.
  uint32_t rebuild_batch = std::max<uint32_t>(rebuild / batch_size, 1);
  uint32_t rehash_batch = std::max<uint32_t>(rehash / batch_size, 1);

  uint64_t num_train_batches = train.NumBatches();

  // Because of how the datasets are read we know that all batches will not have
  // a batch size larger than this so we can just set the batch size here.
  for (uint32_t l = 0; l < _num_layers; l++) {
    _layers[l]->initializeLayer(batch_size);
  }

  ProgressBar bar(num_train_batches);
  for (uint32_t epoch = 0; epoch < epochs; epoch++) {
    bar.reset();
    std::cout << "\nEpoch " << (epoch + 1) << "/" << epochs << ':' << std::endl;
    auto train_start = std::chrono::high_resolution_clock::now();
    for (uint32_t batch = 0; batch < num_train_batches; batch++) {
      if (_iter % 1000 == 999) {
        for (uint32_t i = 0; i < _num_layers; i++) {
          _layers[i]->shuffleRandNeurons();
        }
      }

      processTrainingBatch(train[batch], learning_rate);

      if (_iter % rebuild_batch == (rebuild_batch - 1)) {
        reBuildHashFunctions();
        buildHashTables();
      } else if (_iter % rehash_batch == (rehash_batch - 1)) {
        buildHashTables();
      }

      bar.update();
    }

    auto train_end = std::chrono::high_resolution_clock::now();
    int64_t epoch_time = std::chrono::duration_cast<std::chrono::seconds>(
                             train_end - train_start)
                             .count();
    _time_per_epoch.push_back(epoch_time);
    std::cout << std::endl
              << "Processed " << num_train_batches << " training batches in "
              << epoch_time << " seconds" << std::endl;

    if (intermediate_test_batches == 0) {
      continue;
    }

    uint32_t correct = 0;
    auto test_start = std::chrono::high_resolution_clock::now();
    for (uint32_t batch = 0; batch < intermediate_test_batches; batch++) {
      correct += processTestBatch(test[batch]);
    }
    auto test_end = std::chrono::high_resolution_clock::now();
    std::cout << "Processed " << intermediate_test_batches
              << " test batches in "
              << std::chrono::duration_cast<std::chrono::seconds>(test_end -
                                                                  test_start)
                     .count()
              << " seconds" << std::endl;

    float accuracy = static_cast<float>(correct) / intermediate_test_vecs;
    _accuracy_per_epoch.push_back(accuracy);
    std::cout << "Accuracy: " << accuracy << " (" << correct << "/"
              << intermediate_test_vecs << ")" << std::endl;
  }

  uint64_t num_test_batches = test.NumBatches();
  uint32_t final_correct = 0;
  for (uint32_t batch = 0; batch < num_test_batches; batch++) {
    final_correct += processTestBatch(test[batch]);
  }

  _final_accuracy = static_cast<float>(final_correct) / test.NumVecs();
  std::cout << "Accuracy after training: " << _final_accuracy << " ("
            << final_correct << "/" << test.NumVecs() << ")" << std::endl;
}

void Network::reBuildHashFunctions() {
  for (uint32_t l = 0; l < _num_layers; l++) {
    _layers[l]->reBuildHashFunction();
  }
}

void Network::buildHashTables() {
  for (uint32_t l = 0; l < _num_layers; l++) {
    _layers[l]->buildHashTables();
  }
}

Network::~Network() {
  for (uint32_t i = 0; i < _num_layers; i++) {
    delete _layers[i];
  }

  delete[] _layers;
}

}  // namespace thirdai::bolt