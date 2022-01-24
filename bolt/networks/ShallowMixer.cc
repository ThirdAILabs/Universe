#include "ShallowMixer.h"
#include <bolt/layers/LossFunctions.h>
#include <bolt/utils/ProgressBar.h>
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

ShallowMixer::ShallowMixer(std::vector<FullyConnectedLayerConfig> configs,
                           uint32_t num_patches, uint32_t pixels_per_patch)
    : _configs(std::move(configs)),
      _num_patches(num_patches),
      _pixels_per_patch(pixels_per_patch),
      _iter(0),
      _epoch_count(0) {
  auto start = std::chrono::high_resolution_clock::now();

  _sparse_inference = false;
  _num_layers = _configs.size();
  _layers = new FullyConnectedLayer*[_num_layers];
  _states =
      new BatchState[_num_layers - 1]();  // No stored state for output layer

  std::cout << "====== Building Fully Connected Network ======" << std::endl;

  for (uint32_t i = 0; i < _num_layers; i++) {
    uint64_t prev_dim = (i > 0) ? _configs[i - 1].dim : _pixels_per_patch;
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

std::vector<int64_t> ShallowMixer::train(
    const dataset::InMemoryDataset<dataset::SparseBatch>& train_data,
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

  BatchState outputs = _layers[_num_layers - 1]->createBatchState(batch_size);

  SparseCategoricalCrossEntropyLoss loss;

  for (uint32_t epoch = 0; epoch < epochs; epoch++) {
    std::cout << "\nEpoch " << (_epoch_count + 1) << ':' << std::endl;
    ProgressBar bar(num_train_batches);
    auto train_start = std::chrono::high_resolution_clock::now();

    for (uint32_t batch = 0; batch < num_train_batches; batch++) {
      if (_iter % 1000 == 999) {
        shuffleRandomNeurons();
      }

      const dataset::SparseBatch& input_batch = train_data[batch];

#pragma omp parallel for default(none) shared(input_batch, outputs, loss)
      for (uint32_t i = 0; i < input_batch.getBatchSize(); i++) {
        VectorState input = VectorState::makeSparseInputState(
            input_batch[i]._indices, input_batch[i]._values,
            input_batch[i].length());

        this->forward(i, input, outputs[i], input_batch.labels(i).data(),
                      input_batch.labels(i).size());

        loss(outputs[i], input_batch.getBatchSize(),
             input_batch.labels(i).data(), input_batch.labels(i).size());

        this->backpropagate<true>(i, input, outputs[i]);
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

float ShallowMixer::test(
    const dataset::InMemoryDataset<dataset::SparseBatch>& test_data,
    uint32_t batch_limit) {
  uint32_t batch_size = test_data[0].getBatchSize();

  uint64_t num_test_batches = std::min(test_data.numBatches(), batch_limit);

  // Because of how the datasets are read we know that all batches will not have
  // a batch size larger than this so we can just set the batch size here.
  this->createBatchStates(batch_size, true);

  BatchState outputs = _layers[_num_layers - 1]->createBatchState(
      batch_size, !_layers[_num_layers - 1]->isForceSparsity());

  std::atomic<uint32_t> correct{0};
  ProgressBar bar(num_test_batches);

  auto test_start = std::chrono::high_resolution_clock::now();
  for (uint32_t batch = 0; batch < num_test_batches; batch++) {
    const dataset::SparseBatch& input_batch = test_data[batch];

#pragma omp parallel for default(none) shared(input_batch, outputs, correct)
    for (uint32_t i = 0; i < input_batch.getBatchSize(); i++) {
      VectorState input = VectorState::makeSparseInputState(
          input_batch[i]._indices, input_batch[i]._values,
          input_batch[i].length());

      this->forward(i, input, outputs[i], nullptr, 0);

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

      if (std::find(input_batch.labels(i).begin(), input_batch.labels(i).end(),
                    pred) != input_batch.labels(i).end()) {
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
void ShallowMixer::perPatchForward(uint32_t batch_index,
                                   const VectorState& input) {
  // TODO: Since this is a prototype, assume we are jsut using the first layer
  // for now.
  // TODO: Also need to change the forward and backward routines.
  // TODO: Need to store n_patches and pixels_per_patch privately.
  for (uint32_t patch = 0; patch < _num_patches, ++patch) {
    // TODO: Make dense later
    auto offset = patch * _pixels_per_patch;
    auto patch_input = VectorState::makeSparseInputState(
        input.active_neurons + offset, input.activations + offset,
        _pixels_per_patch);
    // TODO: What to do with this dense thing?
    auto output_vec = _states[0][batch_index];
    // patch_output refers to arrays in _states[0][batch_index]. This makes the
    // layer forward prop to disjoint parts
    auto patch_output = VectorState(
        output_vec.active_neurons + offset, output_vec.activations + offset,
        output_vec.gradients + offset, _pixels_per_patch);
    _layers[0]->forward(patch_input, patch_output);
  }
}

void ShallowMixer::forward(uint32_t batch_index, const VectorState& input,
                           VectorState& output, const uint32_t* labels,
                           uint32_t label_len) {
  // uint32_t n_pixels_in_patch = 256; // TODO: Need permanent solution. I think
  // I have it now? uint32_t n_patches = 64; // TODO: Need permanent solution. I
  // think I have it now?
  for (uint32_t i = 0; i < _num_layers; i++) {
    // There must be at least 2 layers, the first one is the same fully
    // connected layer for each patch
    if (i == 0) {  // First layer
      this->perPatchForward(batch_index, input);
      // _layers[0]->forward(input, _states[0][batch_index],);
    } else if (i == _num_layers - 1) {  // Last layer
      _layers[i]->forward(_states[i - 1][batch_index], output, labels,
                          label_len);
    } else {  // Middle layer
      _layers[i]->forward(_states[i - 1][batch_index], _states[i][batch_index]);
    }
  }
}

template void ShallowMixer::backpropagate<true>(uint32_t, VectorState&,
                                                VectorState&);
template void ShallowMixer::backpropagate<false>(uint32_t, VectorState&,
                                                 VectorState&);

template <bool FROM_INPUT>
void ShallowMixer::backpropagate(uint32_t batch_index, VectorState& input,
                                 VectorState& output) {
  for (uint32_t i = _num_layers; i > 0; i--) {
    uint32_t layer = i - 1;
    // Again we assume at least 2 layers
    if (layer == 0) {  // First layer

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

void ShallowMixer::createBatchStates(uint32_t batch_size, bool force_dense) {
  for (uint32_t l = 0; l < _num_layers - 1; l++) {
    _states[l] = _layers[l]->createBatchState(batch_size, force_dense);
  }
}

void ShallowMixer::updateParameters(float learning_rate) {
  ++_iter;
  for (uint32_t layer = 0; layer < _num_layers; layer++) {
    _layers[layer]->updateParameters(learning_rate, _iter, BETA1, BETA2, EPS);
  }
}

void ShallowMixer::reBuildHashFunctions() {
  for (uint32_t l = 0; l < _num_layers; l++) {
    _layers[l]->reBuildHashFunction();
  }
}

void ShallowMixer::buildHashTables() {
  for (uint32_t l = 0; l < _num_layers; l++) {
    _layers[l]->buildHashTables();
  }
}

void ShallowMixer::shuffleRandomNeurons() {
  for (uint32_t i = 0; i < _num_layers; i++) {
    _layers[i]->shuffleRandNeurons();
  }
}

ShallowMixer::~ShallowMixer() {
  for (uint32_t i = 0; i < _num_layers; i++) {
    delete _layers[i];
  }

  delete[] _layers;
}

}  // namespace thirdai::bolt
