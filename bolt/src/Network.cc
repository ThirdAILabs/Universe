#include "Network.h"
#include <algorithm>
#include <atomic>
#include <chrono>
#include <iostream>
#include <limits>

namespace thirdai::bolt {

Network::Network(std::vector<LayerConfig> configs, uint64_t input_dim)
    : configs(configs), input_dim(input_dim), batch_size_hint(0), iter(0) {
  auto start = std::chrono::high_resolution_clock::now();

  num_layers = configs.size();
  layers = new Layer*[num_layers];

  for (uint32_t i = 0; i < num_layers; i++) {
    uint64_t prev_dim = (i > 0) ? configs[i - 1].dim : input_dim;
    layers[i] = new Layer(configs[i].dim, prev_dim, configs[i].sparsity,
                          configs[i].act_func, configs[i].sampling_config);
  }

  auto end = std::chrono::high_resolution_clock::now();

  std::cout
      << "\033[1;36mInitialized Network in "
      << std::chrono::duration_cast<std::chrono::seconds>(end - start).count()
      << " seconds \033[0m" << std::endl;
}

void Network::ProcessTrainingBatch(const Batch& batch, float lr) {
  uint32_t batch_size = batch.batch_size;
  if (batch_size != batch_size_hint) {
    for (uint32_t layer = 0; layer < num_layers; layer++) {
      layers[layer]->SetBatchSize(batch_size);
    }
    batch_size_hint = batch_size;
  }

#pragma omp parallel for default(none) shared(batch, batch_size)
  for (uint32_t b = 0; b < batch_size; b++) {
    /**
     * 1. Feed Forward
     */
    layers[0]->ForwardPass(b, batch.indices[b], batch.values[b], batch.lens[b]);

    for (uint32_t l = 1; l < num_layers - 1; l++) {
      layers[l]->ForwardPass(b, layers[l - 1]->GetIndices(b),
                             layers[l - 1]->GetValues(b),
                             layers[l - 1]->GetLen(b));
    }
    layers[num_layers - 1]->ForwardPass(
        b, layers[num_layers - 2]->GetIndices(b),
        layers[num_layers - 2]->GetValues(b), layers[num_layers - 2]->GetLen(b),
        batch.labels[b], batch.label_lens[b]);

    /**
     * 2. Compute Errors
     */
    layers[num_layers - 1]->ComputeErrors(b, batch.labels[b],
                                          batch.label_lens[b]);

    /**
     * 3. Backpropogation
     */

    for (uint32_t l = num_layers - 1; l > 0; l--) {
      layers[l]->BackPropagate<false>(
          b, layers[l - 1]->GetIndices(b), layers[l - 1]->GetValues(b),
          layers[l - 1]->GetErrors(b), layers[l - 1]->GetLen(b));
    }

    layers[0]->BackPropagate<true>(b, batch.indices[b], batch.values[b],
                                   nullptr, batch.lens[b]);
  }

  ++iter;
  for (uint32_t layer = 0; layer < num_layers; layer++) {
    layers[layer]->UpdateParameters(lr, iter);
  }
}

uint32_t Network::ProcessTestBatch(const Batch& batch) {
  uint32_t batch_size = batch.batch_size;
  if (batch_size != batch_size_hint) {
    for (uint32_t layer = 0; layer < num_layers; layer++) {
      layers[layer]->SetBatchSize(batch_size);
    }
    batch_size_hint = batch_size;
  }

  for (uint32_t layer = 0; layer < num_layers; layer++) {
    layers[layer]->SetSparsity(1.0);
  }

  std::atomic<uint32_t> correct{0};

#pragma omp parallel for default(none) shared(batch, batch_size, correct)
  for (uint32_t b = 0; b < batch_size; b++) {
    layers[0]->ForwardPass(b, batch.indices[b], batch.values[b], batch.lens[b]);

    for (uint32_t l = 1; l < num_layers - 1; l++) {
      layers[l]->ForwardPass(b, layers[l - 1]->GetIndices(b),
                             layers[l - 1]->GetValues(b),
                             layers[l - 1]->GetLen(b));
    }
    layers[num_layers - 1]->ForwardPass(
        b, layers[num_layers - 2]->GetIndices(b),
        layers[num_layers - 2]->GetValues(b), layers[num_layers - 2]->GetLen(b),
        batch.labels[b], batch.label_lens[b]);

    const uint32_t* indices = layers[num_layers - 1]->GetIndices(b);
    const float* activations = layers[num_layers - 1]->GetValues(b);
    float max_act = std::numeric_limits<float>::min();
    uint32_t pred = 0;
    for (uint32_t i = 0; i < layers[num_layers - 1]->GetLen(b); i++) {
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

  for (uint32_t layer = 0; layer < num_layers; layer++) {
    layers[layer]->SetSparsity(configs[layer].sparsity);
  }

  return correct;
}

void Network::Train(uint32_t batch_size, std::string train_data,
                    std::string test_data, float learning_rate, uint32_t epochs,
                    uint32_t rehash_in, uint32_t rebuild_in,
                    uint32_t max_test_batches) {
  SvmDataset train(train_data, batch_size);
  SvmDataset test(test_data, batch_size);

  uint32_t rehash = rehash_in != 0 ? rehash_in : (train.NumVecs() / 100);
  uint32_t rebuild = rebuild_in != 0 ? rebuild_in : (train.NumVecs() / 4);

  uint64_t intermediate_test_batches =
      std::min<uint64_t>(test.NumBatches(), max_test_batches);
  uint64_t intermediate_test_vecs = std::min<uint64_t>(
      test.NumVecs(), intermediate_test_batches * batch_size);

  if (intermediate_test_batches > 0) {
    uint32_t correct = 0;
    for (uint32_t batch = 0; batch < intermediate_test_batches; batch++) {
      correct += ProcessTestBatch(test[batch]);
    }
    std::cout << "\033[1;32mBefore training accuracy: "
              << ((double)correct / intermediate_test_vecs) << " (" << correct
              << "/" << intermediate_test_vecs << ")\033[0m" << std::endl;
  }

  uint32_t rebuild_batch = rebuild / batch_size;
  uint32_t rehash_batch = rehash / batch_size;

  uint64_t num_train_batches = train.NumBatches();
  uint32_t print = num_train_batches / 10;

  for (uint32_t epoch = 0; epoch < epochs; epoch++) {
    std::cout << "---------|" << std::endl;
    auto train_start = std::chrono::high_resolution_clock::now();
    for (uint32_t batch = 0; batch < num_train_batches; batch++) {
      if (iter % 1000 == 999) {
        for (uint32_t i = 0; i < num_layers; i++) {
          layers[i]->ShuffleRandNeurons();
        }
      }

      ProcessTrainingBatch(train[batch], learning_rate);

      if (iter % rebuild_batch == (rebuild_batch - 1)) {
        ReBuildHashFunctions();
        BuildHashTables();
      } else if (iter % rehash_batch == (rehash_batch - 1)) {
        BuildHashTables();
      }

      if ((batch % print) == (print - 1)) {
        std::cout << "." << std::flush;
      }
    }

    auto train_end = std::chrono::high_resolution_clock::now();
    std::cout << std::endl
              << "\033[1;36mEpoch: " << epoch << "\nProcessed "
              << num_train_batches << " training batches in "
              << std::chrono::duration_cast<std::chrono::seconds>(train_end -
                                                                  train_start)
                     .count()
              << " seconds\033[0m" << std::endl;

    // if (PyErr_CheckSignals() != 0) {
    //   throw py::error_already_set();
    // }

    if (intermediate_test_batches == 0) {
      continue;
    }

    uint32_t correct = 0;
    auto test_start = std::chrono::high_resolution_clock::now();
    for (uint32_t batch = 0; batch < intermediate_test_batches; batch++) {
      correct += ProcessTestBatch(test[batch]);
    }
    auto test_end = std::chrono::high_resolution_clock::now();
    std::cout << "\033[1;36mProcessed " << intermediate_test_batches
              << " test batches in "
              << std::chrono::duration_cast<std::chrono::seconds>(test_end -
                                                                  test_start)
                     .count()
              << " seconds\033[0m" << std::endl;

    std::cout << "\033[1;32mAccuracy: "
              << ((double)correct / intermediate_test_vecs) << " (" << correct
              << "/" << intermediate_test_vecs << ")\033[0m" << std::endl;
  }

  uint64_t num_test_batches = test.NumBatches();
  uint32_t final_correct = 0;
  for (uint32_t batch = 0; batch < num_test_batches; batch++) {
    final_correct += ProcessTestBatch(test[batch]);
  }

  std::cout << "\033[1;32mAfter training accuracy: "
            << ((double)final_correct / test.NumVecs()) << " (" << final_correct
            << "/" << test.NumVecs() << ")\033[0m" << std::endl;
}

uint32_t* Network::PredictClasses(const Batch& batch, uint64_t batch_size) {
  if (batch_size != batch_size_hint) {
    for (uint32_t layer = 0; layer < num_layers; layer++) {
      layers[layer]->SetBatchSize(batch_size);
    }
    batch_size_hint = batch_size;
  }

  for (uint32_t layer = 0; layer < num_layers; layer++) {
    layers[layer]->SetSparsity(1.0);
  }

  uint32_t* predictions = new uint32_t[batch_size];

#pragma omp parallel for default(none) shared(batch, batch_size, predictions)
  for (uint32_t b = 0; b < batch_size; b++) {
    layers[0]->ForwardPass(b, batch.indices[b], batch.values[b], batch.lens[b]);

    for (uint32_t l = 1; l < num_layers - 1; l++) {
      layers[l]->ForwardPass(b, layers[l - 1]->GetIndices(b),
                             layers[l - 1]->GetValues(b),
                             layers[l - 1]->GetLen(b));
    }
    layers[num_layers - 1]->ForwardPass(
        b, layers[num_layers - 2]->GetIndices(b),
        layers[num_layers - 2]->GetValues(b), layers[num_layers - 2]->GetLen(b),
        batch.labels[b], batch.label_lens[b]);

    const uint32_t* indices = layers[num_layers - 1]->GetIndices(b);
    const float* activations = layers[num_layers - 1]->GetValues(b);
    float max_act = std::numeric_limits<float>::min();
    uint32_t pred = 0;
    for (uint32_t i = 0; i < layers[num_layers - 1]->GetLen(b); i++) {
      if (activations[i] > max_act) {
        max_act = activations[i];
        pred = indices[i];
      }
    }

    predictions[b] = pred;
  }

  for (uint32_t layer = 0; layer < num_layers; layer++) {
    layers[layer]->SetSparsity(configs[layer].sparsity);
  }

  return predictions;
}

void Network::ReBuildHashFunctions() {
  for (uint32_t l = 0; l < num_layers; l++) {
    layers[l]->ReBuildHashFunction();
  }
}

void Network::BuildHashTables() {
  for (uint32_t l = 0; l < num_layers; l++) {
    layers[l]->BuildHashTables();
  }
}

Network::~Network() {
  for (uint32_t i = 0; i < num_layers; i++) {
    delete layers[i];
  }

  delete[] layers;
}

}  // namespace thirdai::bolt
