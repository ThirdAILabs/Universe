#pragma once

#include <bolt_vector/src/BoltVector.h>
#include <dataset/src/Datasets.h>
#include <algorithm>
#include <numeric>
#include <random>
#include <unordered_set>

namespace thirdai::bolt::tests {

struct TestDatasetGenerators {
  // Creates a categorical dataset where the inputs are the one hot encoded
  // vectors of the labels, with some small amount of noise added.
  static std::tuple<dataset::BoltDatasetPtr, dataset::BoltDatasetPtr>
  generateSimpleVectorDataset(uint32_t n_classes, uint32_t n_batches,
                              uint32_t batch_size, bool noisy_dataset,
                              bool sparse = false) {
    std::mt19937 gen(892734);
    std::uniform_int_distribution<uint32_t> label_dist(0, n_classes - 1);
    std::normal_distribution<float> data_dist(0, noisy_dataset ? 1.0 : 0.1);

    std::vector<BoltBatch> data_batches;
    std::vector<BoltBatch> label_batches;
    for (uint32_t b = 0; b < n_batches; b++) {
      std::vector<BoltVector> labels;
      std::vector<BoltVector> vectors;
      for (uint32_t i = 0; i < batch_size; i++) {
        uint32_t label = label_dist(gen);
        BoltVector v(/*l= */ n_classes, /* is_dense= */ true,
                     /* has_gradient*/ false);
        std::generate(v.activations, v.activations + n_classes,
                      [&]() { return data_dist(gen); });
        if (!noisy_dataset) {
          v.activations[label] += 1.0;
        }
        if (sparse) {
          auto top_k_indices = v.topKNeurons(n_classes / 2);
          v = BoltVector(/*l= */ top_k_indices.size(), /* is_dense= */ false,
                         /* has_gradient*/ false);
          uint32_t index = 0;
          while (!top_k_indices.empty()) {
            v.active_neurons[index] = top_k_indices.top().second;
            v.activations[index] = top_k_indices.top().first;
            index++;
            top_k_indices.pop();
          }
        }
        vectors.push_back(std::move(v));
        labels.push_back(BoltVector::makeSparseVector({label}, {1.0}));
      }
      data_batches.push_back(BoltBatch(std::move(vectors)));
      label_batches.push_back(BoltBatch(std::move(labels)));
    }

    return std::make_tuple(
        std::make_shared<dataset::BoltDataset>(std::move(data_batches)),
        std::make_shared<dataset::BoltDataset>(std::move(label_batches)));
  }

  // Creates a simple token and vector dataset in which the vector inputs are
  // one hot encoded vectors of the labels with some noise added, and the token
  // inputs are the labels themselves. The noise parameters allow for either one
  // of the inputs to be completely randomized to test that a model can learn
  // just from one of the inputs.
  static std::tuple<dataset::BoltDatasetPtr, dataset::BoltDatasetPtr,
                    dataset::BoltDatasetPtr>
  generateDlrmDataset(uint32_t n_classes, uint32_t n_batches,
                      uint32_t batch_size, bool dense_features_are_noise,
                      bool categorical_features_are_noise, uint32_t seed) {
    std::mt19937 gen(seed);
    std::uniform_int_distribution<uint32_t> label_dist(0, n_classes - 1);
    std::normal_distribution<float> data_dist(
        0, dense_features_are_noise ? 1.0 : 0.1);

    std::vector<BoltBatch> data_batches;
    std::vector<BoltBatch> token_batches;
    std::vector<BoltBatch> label_batches;
    for (uint32_t batch_id = 0; batch_id < n_batches; batch_id++) {
      std::vector<BoltVector> labels;
      std::vector<BoltVector> dense_features;
      std::vector<BoltVector> categorical_features;
      for (uint32_t vec_id = 0; vec_id < batch_size; vec_id++) {
        uint32_t label = label_dist(gen);
        BoltVector v(n_classes, true, false);
        std::generate(v.activations, v.activations + n_classes,
                      [&]() { return data_dist(gen); });
        if (!dense_features_are_noise) {
          v.activations[label] += 1.0;
        }
        dense_features.push_back(std::move(v));
        categorical_features.push_back(BoltVector::singleElementSparseVector(
            categorical_features_are_noise ? label_dist(gen) : label));
        labels.push_back(BoltVector::makeSparseVector({label}, {1.0}));
      }
      data_batches.emplace_back(std::move(dense_features));
      token_batches.emplace_back(std::move(categorical_features));
      label_batches.emplace_back(std::move(labels));
    }

    return {std::make_shared<dataset::BoltDataset>(std::move(data_batches)),
            std::make_shared<dataset::BoltDataset>(std::move(token_batches)),
            std::make_shared<dataset::BoltDataset>(std::move(label_batches))};
  }

  // See comment in DlrmAttentionNodeTests.cc for use of this dataset.
  static std::tuple<dataset::BoltDatasetPtr, dataset::BoltDatasetPtr,
                    dataset::BoltDatasetPtr>
  generateDlrmAttentionDataset(uint32_t n_ids, uint32_t n_tokens,
                               uint32_t n_batches, uint32_t batch_size,
                               uint32_t seed) {
    std::mt19937 gen(seed);
    std::uniform_int_distribution<uint32_t> id_dist(0, n_ids - 1);
    std::normal_distribution<float> data_dist(0, 0.1);

    std::vector<BoltBatch> data_batches;
    std::vector<BoltBatch> token_batches;
    std::vector<BoltBatch> label_batches;
    for (uint32_t batch_id = 0; batch_id < n_batches; batch_id++) {
      std::vector<BoltVector> dense_features;
      std::vector<BoltVector> categorical_features;
      std::vector<BoltVector> labels;
      for (uint32_t vec_id = 0; vec_id < batch_size; vec_id++) {
        uint32_t id = id_dist(gen);

        BoltVector dense_input(n_ids, /* is_dense= */ true,
                               /* has_gradient= */ false);
        std::generate(dense_input.activations, dense_input.activations + n_ids,
                      [&]() { return data_dist(gen); });
        dense_input.activations[id] += 1.0;

        uint32_t id_in_tokens = id_dist(gen) % 2;
        std::unordered_set<uint32_t> tokens;
        if (id_in_tokens) {
          tokens.insert(id);
        }

        while (tokens.size() < n_tokens) {
          uint32_t random_token = id_dist(gen);
          if (random_token != id) {
            tokens.insert(random_token);
          }
        }

        auto token_vector = BoltVector::makeSparseVector(
            std::vector<uint32_t>(tokens.begin(), tokens.end()),
            std::vector<float>(tokens.size(), 1.0));

        BoltVector label_vec(/* l= */ 1, /* is_dense= */ false,
                             /* has_gradient= */ false);
        label_vec.active_neurons[0] = id_in_tokens;
        label_vec.activations[0] = 1.0;

        dense_features.push_back(std::move(dense_input));
        categorical_features.emplace_back(std::move(token_vector));
        labels.push_back(std::move(label_vec));
      }
      data_batches.emplace_back(std::move(dense_features));
      token_batches.emplace_back(std::move(categorical_features));
      label_batches.emplace_back(std::move(labels));
    }

    return {std::make_shared<dataset::BoltDataset>(std::move(data_batches)),
            std::make_shared<dataset::BoltDataset>(std::move(token_batches)),
            std::make_shared<dataset::BoltDataset>(std::move(label_batches))};
  }
};

}  // namespace thirdai::bolt::tests