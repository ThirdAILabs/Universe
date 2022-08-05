#pragma once

#include <dataset/src/Datasets.h>
#include <algorithm>
#include <random>

namespace thirdai::bolt::tests {

struct TestDatasetGenerators {
  static std::tuple<dataset::BoltDatasetPtr, dataset::BoltDatasetPtr>
  generateSimpleVectorDataset(uint32_t n_classes, uint32_t n_batches,
                              uint32_t batch_size, bool noisy_dataset) {
    std::mt19937 gen(892734);
    std::uniform_int_distribution<uint32_t> label_dist(0, n_classes - 1);
    std::normal_distribution<float> data_dist(0, noisy_dataset ? 1.0 : 0.1);

    std::vector<bolt::BoltBatch> data_batches;
    std::vector<bolt::BoltBatch> label_batches;
    for (uint32_t b = 0; b < n_batches; b++) {
      std::vector<bolt::BoltVector> labels;
      std::vector<bolt::BoltVector> vectors;
      for (uint32_t i = 0; i < batch_size; i++) {
        uint32_t label = label_dist(gen);
        bolt::BoltVector v(n_classes, true, false);
        std::generate(v.activations, v.activations + n_classes,
                      [&]() { return data_dist(gen); });
        if (!noisy_dataset) {
          v.activations[label] += 1.0;
        }
        vectors.push_back(std::move(v));
        labels.push_back(BoltVector::makeSparseVector({label}, {1.0}));
      }
      data_batches.push_back(bolt::BoltBatch(std::move(vectors)));
      label_batches.push_back(bolt::BoltBatch(std::move(labels)));
    }

    return std::make_tuple(
        std::make_shared<dataset::BoltDataset>(std::move(data_batches)),
        std::make_shared<dataset::BoltDataset>(std::move(label_batches)));
  }

  // This generates a dataset of random numbers whose label is 0 if even and 1
  // if odd. This tests the embedding layers ability to essentially memorize the
  // dataset.
  static std::tuple<
      std::shared_ptr<dataset::InMemoryDataset<dataset::BoltTokenBatch>>,
      dataset::BoltDatasetPtr>
  generateSimplTokenDataset(uint32_t n_batches, uint32_t batch_size,
                            uint32_t seed) {
    uint32_t dataset_size = n_batches * batch_size;

    std::vector<uint32_t> tokens(dataset_size);

    std::iota(tokens.begin(), tokens.end(), 1);
    std::shuffle(tokens.begin(), tokens.end(), std::mt19937(seed));

    std::vector<dataset::BoltTokenBatch> data;
    std::vector<BoltBatch> labels;

    for (uint32_t batch_index = 0; batch_index < n_batches; batch_index++) {
      std::vector<std::vector<uint32_t>> batch_data;
      std::vector<BoltVector> batch_labels;

      for (uint32_t vec_index = 0; vec_index < batch_size; vec_index++) {
        uint32_t token = tokens[batch_index * batch_size + vec_index];
        batch_data.push_back({token});
        batch_labels.push_back(
            BoltVector::makeSparseVector({token % 2}, {1.0}));
      }

      data.push_back(dataset::BoltTokenBatch(std::move(batch_data)));
      labels.push_back(BoltBatch(std::move(batch_labels)));
    }

    return std::make_pair(
        std::make_shared<dataset::InMemoryDataset<dataset::BoltTokenBatch>>(
            std::move(data)),
        std::make_shared<dataset::BoltDataset>(std::move(labels)));
  }

  static std::tuple<dataset::BoltDatasetPtr, dataset::BoltTokenDatasetPtr,
                    dataset::BoltDatasetPtr>
  generateDlrmDataset(uint32_t n_classes, uint32_t n_batches,
                      uint32_t batch_size, bool dense_features_are_noise,
                      bool categorical_features_are_noise, uint32_t seed) {
    std::mt19937 gen(seed);
    std::uniform_int_distribution<uint32_t> label_dist(0, n_classes - 1);
    std::normal_distribution<float> data_dist(
        0, dense_features_are_noise ? 1.0 : 0.1);

    std::vector<BoltBatch> data_batches;
    std::vector<dataset::BoltTokenBatch> token_batches;
    std::vector<BoltBatch> label_batches;
    for (uint32_t batch_id = 0; batch_id < n_batches; batch_id++) {
      std::vector<BoltVector> labels;
      std::vector<BoltVector> dense_features;
      std::vector<std::vector<uint32_t>> categorical_features;
      for (uint32_t vec_id = 0; vec_id < batch_size; vec_id++) {
        uint32_t label = label_dist(gen);
        BoltVector v(n_classes, true, false);
        std::generate(v.activations, v.activations + n_classes,
                      [&]() { return data_dist(gen); });
        if (!dense_features_are_noise) {
          v.activations[label] += 1.0;
        }
        dense_features.push_back(std::move(v));
        categorical_features.push_back(
            {categorical_features_are_noise ? label_dist(gen) : label});
        labels.push_back(BoltVector::makeSparseVector({label}, {1.0}));
      }
      data_batches.emplace_back(std::move(dense_features));
      token_batches.emplace_back(std::move(categorical_features));
      label_batches.emplace_back(std::move(labels));
    }

    return {
        std::make_shared<dataset::BoltDataset>(std::move(data_batches)),
        std::make_shared<dataset::BoltTokenDataset>(std::move(token_batches)),
        std::make_shared<dataset::BoltDataset>(std::move(label_batches))};
  }
};

}  // namespace thirdai::bolt::tests