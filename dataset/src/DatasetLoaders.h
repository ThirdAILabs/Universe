#pragma once

#include "BatchProcessor.h"
#include "Datasets.h"
#include "InMemoryDataset.h"
#include "dataset/src/dataset_loaders/TabularDatasetLoader.h"
#include <bolt_vector/src/BoltVector.h>
#include <dataset/src/batch_processors/ClickThroughBatchProcessor.h>
#include <dataset/src/batch_processors/SvmBatchProcessor.h>
#include <memory>
#include <utility>

namespace thirdai::dataset {

struct SvmDatasetLoader {
  static std::tuple<BoltDatasetPtr, BoltDatasetPtr> loadDatasetFromFile(
      const std::string& filename, uint32_t batch_size,
      bool softmax_for_multiclass = true) {
    auto data_source =
        std::make_shared<SimpleFileDataSource>(filename, batch_size);
    return loadDataset(data_source, softmax_for_multiclass);
  }

  static std::tuple<BoltDatasetPtr, BoltDatasetPtr> loadDataset(
      const std::shared_ptr<DataSource>& data_source,
      bool softmax_for_multiclass = true) {
    auto batch_processor =
        std::make_shared<SvmBatchProcessor>(softmax_for_multiclass);
    auto dataset_loader = TabularDatasetLoader(data_source, batch_processor,
                                               /* shuffle = */ false);
    auto datasets = dataset_loader.loadInMemory();
    return {datasets.first.at(0), datasets.second};
  }
};

struct ClickThroughDatasetLoader {
  static std::tuple<BoltDatasetPtr, BoltDatasetPtr, BoltDatasetPtr>
  loadDatasetFromFile(const std::string& filename, uint32_t batch_size,
                      uint32_t num_dense_features,
                      uint32_t max_num_categorical_features, char delimiter) {
    auto data_source =
        std::make_shared<SimpleFileDataSource>(filename, batch_size);
    return loadDataset(data_source, num_dense_features,
                       max_num_categorical_features, delimiter);
  }

  static std::tuple<BoltDatasetPtr, BoltDatasetPtr, BoltDatasetPtr> loadDataset(
      const std::shared_ptr<DataSource>& data_source,
      uint32_t num_dense_features, uint32_t max_num_categorical_features,
      char delimiter) {
    auto batch_processor = std::make_shared<ClickThroughBatchProcessor>(
        num_dense_features, max_num_categorical_features, delimiter);
    auto dataset_loader = TabularDatasetLoader(data_source, batch_processor,
                                               /* shuffle = */ false);
    auto datasets = dataset_loader.loadInMemory();
    return {datasets.first.at(0), datasets.first.at(1), datasets.second};
  }
};

}  // namespace thirdai::dataset