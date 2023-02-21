#pragma once

#include "Datasets.h"
#include "Featurizer.h"
#include "InMemoryDataset.h"
#include <bolt_vector/src/BoltVector.h>
#include <dataset/src/dataset_loaders/DatasetLoader.h>
#include <dataset/src/featurizers/ClickThroughFeaturizer.h>
#include <dataset/src/featurizers/SvmFeaturizer.h>
#include <memory>
#include <unordered_map>
#include <utility>

namespace thirdai::dataset {

struct SvmDatasetLoader {
  static std::tuple<BoltDatasetPtr, BoltDatasetPtr> loadDatasetFromFile(
      const std::string& filename, uint32_t batch_size,
      bool softmax_for_multiclass = true) {
    auto data_source = std::make_shared<FileDataSource>(filename);
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

  static BoltVector toSparseVector(const MapInput& keys_to_values) {
    BoltVector vector(/* l = */ keys_to_values.size(), /* is_dense = */ false);
    size_t current_index = 0;
    for (const auto& key_and_value : keys_to_values) {
      uint32_t key = std::stoul(key_and_value.first);
      float value = std::stof(key_and_value.second);
      vector.active_neurons[current_index] = key;
      vector.activations[current_index] = value;
    }
    return vector;
  }

  static BoltBatch toSparseVectors(const MapInputBatch& keys_to_values_batch) {
    std::vector<BoltVector> sparse_vectors;
    for (const MapInput& keys_to_values : keys_to_values_batch) {
      sparse_vectors.push_back(toSparseVector(keys_to_values));
    }
    return BoltBatch(std::move(sparse_vectors));
  }
};

struct ClickThroughDatasetLoader {
  static std::tuple<BoltDatasetPtr, BoltDatasetPtr, BoltDatasetPtr>
  loadDatasetFromFile(const std::string& filename, uint32_t batch_size,
                      uint32_t num_dense_features,
                      uint32_t max_num_categorical_features, char delimiter) {
    auto data_source = std::make_shared<FileDataSource>(filename);
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