#include "Network.h"
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

Network::Network(std::vector<LayerConfig> configs, uint64_t input_dim)
    : _configs(std::move(configs)),
      _input_dim(input_dim),
      _batch_size_hint(0),
      _iter(0) {
  auto start = std::chrono::high_resolution_clock::now();

  _num_layers = _configs.size();
  _layers = new SparseLayer*[_num_layers];

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
    _layers[i] =
        new SparseLayer(_configs[i].dim, prev_dim, _configs[i].sparsity,
                        _configs[i].act_func, _configs[i].sampling_config);
  }

  auto end = std::chrono::high_resolution_clock::now();

  std::cout
      << "Initialized Network in "
      << std::chrono::duration_cast<std::chrono::seconds>(end - start).count()
      << " seconds" << std::endl;
  std::cout << "==============================" << std::endl;
}

void Network::ProcessTrainingBatch(const Batch& batch, float lr) {
  uint32_t batch_size = batch.batch_size;
  if (batch_size != _batch_size_hint) {
    for (uint32_t layer = 0; layer < _num_layers; layer++) {
      _layers[layer]->SetBatchSize(batch_size);
    }
    _batch_size_hint = batch_size;
  }

#pragma omp parallel for default(none) shared(batch, batch_size)
  for (uint32_t b = 0; b < batch_size; b++) {
    /**
     * 1. Feed Forward
     */
    _layers[0]->FeedForward(b, batch.indices[b], batch.values[b], batch.lens[b],
                            nullptr, 0);

    for (uint32_t l = 1; l < _num_layers - 1; l++) {
      _layers[l]->FeedForward(b, _layers[l - 1]->GetIndices(b),
                              _layers[l - 1]->GetValues(b),
                              _layers[l - 1]->GetLen(b), nullptr, 0);
    }
    _layers[_num_layers - 1]->FeedForward(
        b, _layers[_num_layers - 2]->GetIndices(b),
        _layers[_num_layers - 2]->GetValues(b),
        _layers[_num_layers - 2]->GetLen(b), batch.labels[b],
        batch.label_lens[b]);

    /**
     * 2. Compute Errors
     */
    _layers[_num_layers - 1]->ComputeErrors(b, batch.labels[b],
                                            batch.label_lens[b]);

    /**
     * 3. Backpropogation
     */

    for (uint32_t l = _num_layers - 1; l > 0; l--) {
      _layers[l]->Backpropagate(
          b, _layers[l - 1]->GetIndices(b), _layers[l - 1]->GetValues(b),
          _layers[l - 1]->GetErrors(b), _layers[l - 1]->GetLen(b));
    }

    _layers[0]->BackpropagateFirstLayer(b, batch.indices[b], batch.values[b],
                                        nullptr, batch.lens[b]);
  }

  ++_iter;
  for (uint32_t layer = 0; layer < _num_layers; layer++) {
    _layers[layer]->UpdateParameters(lr, _iter, BETA1, BETA2, EPS);
  }
}

uint32_t Network::ProcessTestBatch(const Batch& batch) {
  uint32_t batch_size = batch.batch_size;
  if (batch_size != _batch_size_hint) {
    for (uint32_t layer = 0; layer < _num_layers; layer++) {
      _layers[layer]->SetBatchSize(batch_size);
    }
    _batch_size_hint = batch_size;
  }

  for (uint32_t layer = 0; layer < _num_layers; layer++) {
    _layers[layer]->SetSparsity(1.0);
  }

  std::atomic<uint32_t> correct{0};

#pragma omp parallel for default(none) shared(batch, batch_size, correct)
  for (uint32_t b = 0; b < batch_size; b++) {
    _layers[0]->FeedForward(b, batch.indices[b], batch.values[b], batch.lens[b],
                            nullptr, 0);

    for (uint32_t l = 1; l < _num_layers - 1; l++) {
      _layers[l]->FeedForward(b, _layers[l - 1]->GetIndices(b),
                              _layers[l - 1]->GetValues(b),
                              _layers[l - 1]->GetLen(b), nullptr, 0);
    }
    _layers[_num_layers - 1]->FeedForward(
        b, _layers[_num_layers - 2]->GetIndices(b),
        _layers[_num_layers - 2]->GetValues(b),
        _layers[_num_layers - 2]->GetLen(b), batch.labels[b],
        batch.label_lens[b]);

    const uint32_t* indices = _layers[_num_layers - 1]->GetIndices(b);
    const float* activations = _layers[_num_layers - 1]->GetValues(b);
    float max_act = std::numeric_limits<float>::min();
    uint32_t pred = 0;
    for (uint32_t i = 0; i < _layers[_num_layers - 1]->GetLen(b); i++) {
      if (activations[i] > max_act) {
        max_act = activations[i];
        pred = indices[i];
      }
    }

    if (std::find(batch.labels[b], batch.labels[b] + batch.label_lens[b],
                  pred) != batch.labels[b] + batch.label_lens[b]) {
      correct++;
    }
  }

  for (uint32_t layer = 0; layer < _num_layers; layer++) {
    _layers[layer]->SetSparsity(_configs[layer].sparsity);
  }

  return correct;
}

void Network::Train(uint32_t batch_size, const std::string& train_data,
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

  uint32_t rebuild_batch = rebuild / batch_size;
  uint32_t rehash_batch = rehash / batch_size;

  uint64_t num_train_batches = train.NumBatches();
  uint32_t print = num_train_batches / 10;

  for (uint32_t epoch = 0; epoch < epochs; epoch++) {
    std::cout << "---------|" << std::endl;
    auto train_start = std::chrono::high_resolution_clock::now();
    for (uint32_t batch = 0; batch < num_train_batches; batch++) {
      if (_iter % 1000 == 999) {
        for (uint32_t i = 0; i < _num_layers; i++) {
          _layers[i]->ShuffleRandNeurons();
        }
      }

      ProcessTrainingBatch(train[batch], learning_rate);

      if (_iter % rebuild_batch == (rebuild_batch - 1)) {
        ReBuildHashFunctions();
        BuildHashTables();
      } else if (_iter % rehash_batch == (rehash_batch - 1)) {
        BuildHashTables();
      }

      if ((batch % print) == (print - 1)) {
        std::cout << "." << std::flush;
      }
    }

    auto train_end = std::chrono::high_resolution_clock::now();
    int64_t epoch_time = std::chrono::duration_cast<std::chrono::seconds>(
                             train_end - train_start)
                             .count();
    _time_per_epoch.push_back(epoch_time);
    std::cout << std::endl
              << "Epoch: " << epoch << "\nProcessed " << num_train_batches
              << " training batches in " << epoch_time << " seconds"
              << std::endl;

    if (intermediate_test_batches == 0) {
      continue;
    }

    uint32_t correct = 0;
    auto test_start = std::chrono::high_resolution_clock::now();
    for (uint32_t batch = 0; batch < intermediate_test_batches; batch++) {
      correct += ProcessTestBatch(test[batch]);
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
    final_correct += ProcessTestBatch(test[batch]);
  }

  _final_accuracy = static_cast<float>(final_correct) / test.NumVecs();
  std::cout << "Accuracy after training: " << _final_accuracy << " ("
            << final_correct << "/" << test.NumVecs() << ")" << std::endl;
}

uint32_t* Network::PredictClasses(const Batch& batch, uint64_t batch_size) {
  if (batch_size != _batch_size_hint) {
    for (uint32_t layer = 0; layer < _num_layers; layer++) {
      _layers[layer]->SetBatchSize(batch_size);
    }
    _batch_size_hint = batch_size;
  }

  for (uint32_t layer = 0; layer < _num_layers; layer++) {
    _layers[layer]->SetSparsity(1.0);
  }

  uint32_t* predictions = new uint32_t[batch_size];

#pragma omp parallel for default(none) shared(batch, batch_size, predictions)
  for (uint32_t b = 0; b < batch_size; b++) {
    _layers[0]->FeedForward(b, batch.indices[b], batch.values[b], batch.lens[b],
                            nullptr, 0);

    for (uint32_t l = 1; l < _num_layers - 1; l++) {
      _layers[l]->FeedForward(b, _layers[l - 1]->GetIndices(b),
                              _layers[l - 1]->GetValues(b),
                              _layers[l - 1]->GetLen(b), nullptr, 0);
    }
    _layers[_num_layers - 1]->FeedForward(
        b, _layers[_num_layers - 2]->GetIndices(b),
        _layers[_num_layers - 2]->GetValues(b),
        _layers[_num_layers - 2]->GetLen(b), batch.labels[b],
        batch.label_lens[b]);

    const uint32_t* indices = _layers[_num_layers - 1]->GetIndices(b);
    const float* activations = _layers[_num_layers - 1]->GetValues(b);
    float max_act = std::numeric_limits<float>::min();
    uint32_t pred = 0;
    for (uint32_t i = 0; i < _layers[_num_layers - 1]->GetLen(b); i++) {
      if (activations[i] > max_act) {
        max_act = activations[i];
        pred = indices[i];
      }
    }

    predictions[b] = pred;
  }

  for (uint32_t layer = 0; layer < _num_layers; layer++) {
    _layers[layer]->SetSparsity(_configs[layer].sparsity);
  }

  return predictions;
}

void Network::ReBuildHashFunctions() {
  for (uint32_t l = 0; l < _num_layers; l++) {
    _layers[l]->ReBuildHashFunction();
  }
}

void Network::BuildHashTables() {
  for (uint32_t l = 0; l < _num_layers; l++) {
    _layers[l]->BuildHashTables();
  }
}

Network::~Network() {
  for (uint32_t i = 0; i < _num_layers; i++) {
    delete _layers[i];
  }

  delete[] _layers;
}

std::ostream& operator<<(std::ostream& out, const LayerConfig& config) {
  out << "Layer: dim=" << config.dim << ", load_factor=" << config.sparsity;
  switch (config.act_func) {
    case ActivationFunc::ReLU:
      out << ", act_func=ReLU";
      break;
    case ActivationFunc::Softmax:
      out << ", act_func=Softmax";
      break;
  }
  if (config.sparsity < 1.0) {
    out << ", sampling: {";
    out << "hashes_per_table=" << config.sampling_config.hashes_per_table
        << ", num_tables=" << config.sampling_config.num_tables
        << ", range_pow=" << config.sampling_config.range_pow
        << ", reservoir_size=" << config.sampling_config.reservoir_size << "}";
  }
  return out;
}

}  // namespace thirdai::bolt
