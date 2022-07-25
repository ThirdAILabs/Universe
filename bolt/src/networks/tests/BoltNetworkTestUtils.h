#pragma once

#include <dataset/src/Datasets.h>
#include <algorithm>
#include <optional>
#include <random>
#include <vector>

namespace thirdai::bolt::tests {

static std::tuple<dataset::BoltDatasetPtr, dataset::BoltDatasetPtr> genDataset(
    uint32_t n_classes, bool noisy_dataset, uint32_t n_batches = 100,
    uint32_t batch_size = 100) {
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

}  // namespace thirdai::bolt::tests