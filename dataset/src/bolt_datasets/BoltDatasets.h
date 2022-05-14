#pragma once

#include <bolt/src/layers/BoltVector.h>
#include <dataset/src/Dataset.h>

namespace thirdai::dataset {

using BoltDataset = InMemoryDataset<bolt::BoltBatch>;

class DatasetWithLabels {
 public:
  BoltDataset data;
  BoltDataset labels;

  explicit DatasetWithLabels(BoltDataset&& _data, BoltDataset&& _labels)
      : data(std::move(_data)), labels(std::move(_labels)) {}
};

DatasetWithLabels loadBoltSvmDataset(const std::string& filename,
                                     uint32_t batch_size);

DatasetWithLabels loadBoltCsvDataset(const std::string& filename,
                                     uint32_t batch_size, char delimiter);

class ClickThroughDatasetWithLabels {
 public:
  InMemoryDataset<ClickThroughBatch> data;
  BoltDataset labels;

  explicit ClickThroughDatasetWithLabels(
      InMemoryDataset<ClickThroughBatch>&& _data, BoltDataset&& _labels)
      : data(std::move(_data)), labels(std::move(_labels)) {}
};

ClickThroughDatasetWithLabels loadClickThroughDataset(
    const std::string& filename, uint32_t batch_size,
    uint32_t num_dense_features, uint32_t num_categorical_features,
    bool sparse_labels);

};  // namespace thirdai::dataset
