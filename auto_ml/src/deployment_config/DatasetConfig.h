#pragma once

#include "BlockConfig.h"
#include <bolt/src/graph/nodes/Input.h>
#include <bolt_vector/src/BoltVector.h>
#include <dataset/src/Datasets.h>
#include <dataset/src/StreamingDataset.h>
#include <dataset/src/StreamingGenericDatasetLoader.h>
#include <dataset/src/batch_processors/GenericBatchProcessor.h>
#include <dataset/src/blocks/BlockInterface.h>
#include <memory>
#include <optional>

namespace thirdai::automl::deployment_config {

using InputDatasets = std::vector<dataset::BoltDatasetPtr>;
using LabelDataset = dataset::BoltDatasetPtr;

class DatasetLoader {
 public:
  virtual std::optional<std::pair<InputDatasets, LabelDataset>> loadInMemory(
      uint32_t max_in_memory_batches) = 0;

  virtual void restart() = 0;

  virtual ~DatasetLoader() = default;
};

class GenericDatasetLoader final : public DatasetLoader {
 public:
  GenericDatasetLoader(std::shared_ptr<dataset::DataLoader> data_loader,
                       dataset::GenericBatchProcessorPtr batch_processor,
                       bool shuffle)
      : _dataset(std::move(data_loader), std::move(batch_processor), shuffle) {}

  std::optional<std::pair<InputDatasets, LabelDataset>> loadInMemory(
      uint32_t max_in_memory_batches) final {
    (void)max_in_memory_batches;
    // TODO(Nicholas, Geordie): add method to load N batches in
    // StreamingGenericDatasetLoader
    auto [data, labels] = _dataset.loadInMemory();

    return std::make_optional<std::pair<InputDatasets, LabelDataset>>(
        InputDatasets{data}, labels);
  }

  void restart() final { _dataset.restart(); }

 private:
  dataset::StreamingGenericDatasetLoader _dataset;
};

using DatasetLoaderPtr = std::unique_ptr<DatasetLoader>;

class DatasetState {
 public:
  virtual void preprocessDataset(
      const std::shared_ptr<dataset::DataLoader>& data_loader,
      std::optional<uint64_t> max_in_memory_batches) {
    (void)data_loader;
    (void)max_in_memory_batches;
  }

  virtual DatasetLoaderPtr getDatasetLoader(
      std::shared_ptr<dataset::DataLoader> data_loader) = 0;

  virtual std::vector<BoltVector> featurizeInput(const std::string& input) = 0;

  virtual std::vector<bolt::InputPtr> getInputNodes() = 0;

  virtual ~DatasetState() = default;
};

using DatasetStatePtr = std::unique_ptr<DatasetState>;

class DatasetConfig {
 public:
  virtual DatasetStatePtr createDatasetState(
      const std::optional<std::string>& option,
      const UserInputMap& user_specified_parameters) const = 0;

  virtual ~DatasetConfig() = default;
};

using DatasetConfigPtr = std::shared_ptr<DatasetConfig>;

}  // namespace thirdai::automl::deployment_config