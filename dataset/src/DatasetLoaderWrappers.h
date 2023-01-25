#pragma once

#include "Datasets.h"
#include "Featurizer.h"
#include "InMemoryDataset.h"
#include <bolt_vector/src/BoltVector.h>
#include <dataset/src/dataset_loaders/DatasetLoader.h>
#include <dataset/src/featurizers/ClickThroughFeaturizer.h>
#include <dataset/src/featurizers/SvmFeaturizer.h>
#include <memory>
#include <utility>

namespace thirdai::dataset {

struct SvmDatasetLoader {
  static std::tuple<BoltDatasetPtr, BoltDatasetPtr> loadDatasetFromFile(
      const std::string& filename, uint32_t batch_size,
      bool softmax_for_multiclass = true) {
    auto data_source = std::make_shared<SimpleFileDataSource>(filename);
    return loadDataset(data_source, batch_size, softmax_for_multiclass);
  }

  static std::tuple<BoltDatasetPtr, BoltDatasetPtr> loadDataset(
      const DataSourcePtr& data_source, size_t batch_size,
      bool softmax_for_multiclass = true) {
    auto featurizer = std::make_shared<SvmFeaturizer>(softmax_for_multiclass);
    auto dataset_loader = DatasetLoader(data_source, featurizer,
                                        /* shuffle = */ false);
    auto datasets = dataset_loader.loadAll(batch_size);
    return {datasets.first.at(0), datasets.second};
  }
};

struct ClickThroughDatasetLoader {
  static std::tuple<BoltDatasetPtr, BoltDatasetPtr, BoltDatasetPtr>
  loadDatasetFromFile(const std::string& filename, uint32_t batch_size,
                      uint32_t num_dense_features,
                      uint32_t max_num_categorical_features, char delimiter) {
    auto data_source = std::make_shared<SimpleFileDataSource>(filename);
    return loadDataset(data_source, num_dense_features,
                       max_num_categorical_features, delimiter, batch_size);
  }

  static std::tuple<BoltDatasetPtr, BoltDatasetPtr, BoltDatasetPtr> loadDataset(
      const DataSourcePtr& data_source, uint32_t num_dense_features,
      uint32_t max_num_categorical_features, char delimiter,
      size_t batch_size) {
    auto featurizer = std::make_shared<ClickThroughFeaturizer>(
        num_dense_features, max_num_categorical_features, delimiter);
    auto dataset_loader = DatasetLoader(data_source, featurizer,
                                        /* shuffle = */ false);
    auto datasets = dataset_loader.loadAll(batch_size);
    return {datasets.first.at(0), datasets.first.at(1), datasets.second};
  }
};

}  // namespace thirdai::dataset