#pragma once

#include <bolt/src/layers/BoltVector.h>
#include <dataset/src/Dataset.h>

namespace thirdai::dataset {

using BoltDataset = InMemoryDataset<bolt::BoltBatch>;
using BoltDatasetPtr = std::shared_ptr<BoltDataset>;

class DatasetWithLabels {
 public:
  BoltDatasetPtr data;
  BoltDatasetPtr labels;

  explicit DatasetWithLabels(BoltDataset&& _data, BoltDataset&& _labels)
      : data(std::make_shared<BoltDataset>(std::move(_data))),
        labels(std::make_shared<BoltDataset>(std::move(_labels))) {}
};

DatasetWithLabels loadBoltSvmDataset(const std::string& filename,
                                     uint32_t batch_size);

DatasetWithLabels loadBoltCsvDataset(const std::string& filename,
                                     uint32_t batch_size, char delimiter);

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
