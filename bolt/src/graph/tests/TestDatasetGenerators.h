#pragma once

#include <bolt_vector/src/BoltVector.h>
#include <dataset/src/Datasets.h>
#include <algorithm>
#include <numeric>
#include <random>

namespace thirdai::bolt::tests {

struct TestDatasetGenerators {
  // Creates a categorical dataset where the inputs are the one hot encoded
  // vectors of the labels, with some small amount of noise added.
  static std::tuple<dataset::BoltDatasetPtr, dataset::BoltDatasetPtr>
  generateSimpleVectorDataset(uint32_t n_classes, uint32_t n_batches,
                              uint32_t batch_size, bool noisy_dataset) {
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
        BoltVector v(n_classes, true, false);
        std::generate(v.activations, v.activations + n_classes,
                      [&]() { return data_dist(gen); });
        if (!noisy_dataset) {
          v.activations[label] += 1.0;
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

  // This generates a dataset of random numbers whose label is 0 if even and 1
  // if odd. This tests the embedding layers ability to essentially memorize the
  // dataset.
  static std::tuple<dataset::BoltDatasetPtr, dataset::BoltDatasetPtr>
  generateSimpleTokenDataset(uint32_t n_batches, uint32_t batch_size,
                             uint32_t seed) {
    uint32_t dataset_size = n_batches * batch_size;

    std::vector<uint32_t> tokens(dataset_size);

    std::iota(tokens.begin(), tokens.end(), 1);
    std::shuffle(tokens.begin(), tokens.end(), std::mt19937(seed));

    std::vector<BoltBatch> data;
    std::vector<BoltBatch> labels;

    for (uint32_t batch_index = 0; batch_index < n_batches; batch_index++) {
      std::vector<BoltVector> batch_data;
      std::vector<BoltVector> batch_labels;

      for (uint32_t vec_index = 0; vec_index < batch_size; vec_index++) {
        uint32_t token = tokens[batch_index * batch_size + vec_index];
        batch_data.push_back(BoltVector::singleElementSparseVector(token));
        batch_labels.push_back(
            BoltVector::singleElementSparseVector(token % 2));
      }

      data.push_back(BoltBatch(std::move(batch_data)));
      labels.push_back(BoltBatch(std::move(batch_labels)));
    }

    return std::make_pair(
        std::make_shared<dataset::BoltDataset>(std::move(data)),
        std::make_shared<dataset::BoltDataset>(std::move(labels)));
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
};

}  // namespace thirdai::bolt::tests