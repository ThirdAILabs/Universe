#pragma once

#include <bolt/src/layers/BoltVector.h>
#include <dataset/src/Dataset.h>
#include <dataset/src/batch_types/BoltTokenBatch.h>

namespace thirdai::dataset {

using BoltDataset = InMemoryDataset<bolt::BoltBatch>;
using BoltDatasetPtr = std::shared_ptr<BoltDataset>;
using BoltDatasetList = std::vector<BoltDatasetPtr>;
using BoltTokenDataset = InMemoryDataset<BoltTokenBatch>;
using BoltTokenDatasetPtr = std::shared_ptr<BoltTokenDataset>;
using BoltTokenDatasetList = std::vector<BoltTokenDatasetPtr>;

class DatasetWithLabels {
 public:
  BoltDatasetPtr data;
  BoltDatasetPtr labels;

  DatasetWithLabels() : data(nullptr), labels(nullptr) {}

  explicit DatasetWithLabels(BoltDataset&& _data, BoltDataset&& _labels)
      : data(std::make_shared<BoltDataset>(std::move(_data))),
        labels(std::make_shared<BoltDataset>(std::move(_labels))) {}
};

std::tuple<BoltDatasetPtr, BoltDatasetPtr> loadBoltSvmDataset(
    const std::string& filename, uint32_t batch_size,
    bool softmax_for_multiclass = true);

using ClickThroughDataset = InMemoryDataset<ClickThroughBatch>;
using ClickThroughDatasetPtr = std::shared_ptr<ClickThroughDataset>;

class ClickThroughDatasetWithLabels {
 public:
  ClickThroughDatasetPtr data;
  BoltDatasetPtr labels;

  explicit ClickThroughDatasetWithLabels(
      InMemoryDataset<ClickThroughBatch>&& _data, BoltDataset&& _labels)
      : data(std::make_shared<InMemoryDataset<ClickThroughBatch>>(
            std::move(_data))),
        labels(std::make_shared<BoltDataset>(std::move(_labels))) {}
};

ClickThroughDatasetWithLabels loadClickThroughDataset(
    const std::string& filename, uint32_t batch_size,
    uint32_t num_dense_features, uint32_t num_categorical_features,
    bool sparse_labels);

}  // namespace thirdai::dataset
