#pragma once

#include <cereal/access.hpp>
#include "BlockConfig.h"
#include <bolt/src/graph/nodes/Input.h>
#include <bolt_vector/src/BoltVector.h>
#include <_types/_uint32_t.h>
#include <auto_ml/src/Aliases.h>
#include <auto_ml/src/deployment_config/Artifact.h>
#include <dataset/src/Datasets.h>
#include <dataset/src/StreamingDataset.h>
#include <dataset/src/StreamingGenericDatasetLoader.h>
#include <dataset/src/batch_processors/GenericBatchProcessor.h>
#include <dataset/src/blocks/BlockInterface.h>
#include <__tuple>
#include <cstdint>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <variant>

namespace thirdai::automl::deployment {

/**
 * Structure of Dataset Configuration and Loading:
 *
 * DatasetConfig:
        A config that specifies how to create the DatasetLoaderFactory.
 *
 * DatasetLoaderFactory:
 *      takes in DataLoaders, for instance S3, a file, etc. and returns a
 *      DatasetLoader for the given resource. This factory can also maintain any
 *      state that's needed to load datasets, for instance in the Tabular and
 *      Sequential data loaders which are stateful.
 *
 * DatasetLoader:
 *      Can return bolt datasets form the associated data source until it is
 *      exhausted. For instance a data loader would be returned by the factory
 *      for each data source train or evaluate is invoked with.
 *
 */

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
    auto datasets = _dataset.loadInMemoryWithMaxBatches(max_in_memory_batches);
    if (!datasets) {
      return std::nullopt;
    }

    auto& [data, labels] = datasets.value();

    return std::make_optional<std::pair<InputDatasets, LabelDataset>>(
        InputDatasets{data}, labels);
  }

  void restart() final { _dataset.restart(); }

 private:
  dataset::StreamingGenericDatasetLoader _dataset;
};

using DatasetLoaderPtr = std::unique_ptr<DatasetLoader>;

class DatasetLoaderFactory {
 public:
  /**
   * Note that preprocess data is called at the begining of train before
   * getLabeldDatasetLoader. It is the responsibility of the implementation to
   * ensure that it maintains the state correctly if called multiple times.
   */
  virtual void preprocessDataset(
      const std::shared_ptr<dataset::DataLoader>& data_loader,
      std::optional<uint64_t> max_in_memory_batches) {
    (void)data_loader;
    (void)max_in_memory_batches;
  }

  virtual DatasetLoaderPtr getLabeledDatasetLoader(
      std::shared_ptr<dataset::DataLoader> data_loader, bool training) = 0;

  virtual std::vector<BoltVector> featurizeInput(const std::string& input) = 0;

  virtual std::vector<BoltVector> featurizeInput(const MapInput& input) {
    (void)input;
    throw std::invalid_argument(
        "This model pipeline configuration does not support map input. Pass in "
        "a string instead.");
  };

  virtual std::vector<BoltBatch> featurizeInputBatch(
      const std::vector<std::string>& inputs) = 0;

  virtual std::vector<BoltBatch> featurizeInputBatch(
      const MapInputBatch& inputs) {
    (void)inputs;
    throw std::invalid_argument(
        "This model pipeline configuration does not support map input. Pass in "
        "a list of strings instead.");
  };

  virtual uint32_t labelToNeuronId(
      std::variant<uint32_t, std::string> label) = 0;

  virtual std::vector<dataset::Explanation> explain(
      const std::optional<std::vector<uint32_t>>& gradients_indices,
      const std::vector<float>& gradients_ratio, const std::string& sample) = 0;

  virtual std::vector<dataset::Explanation> explain(
      const std::optional<std::vector<uint32_t>>& gradients_indices,
      const std::vector<float>& gradients_ratio, const MapInput& sample) {
    (void)gradients_indices;
    (void)gradients_ratio;
    (void)sample;
    throw std::invalid_argument(
        "This model pipeline configuration does not support map input. Pass in "
        "a list of strings instead.");
  }

  virtual std::vector<bolt::InputPtr> getInputNodes() = 0;

  virtual uint32_t getLabelDim() = 0;

  virtual std::optional<std::vector<std::string>> getIdToLabelMap() const = 0;

  virtual ~DatasetLoaderFactory() = default;

  Artifact getArtifact(const std::string& name) const {
    if (auto artifact = getArtifactImpl(name)) {
      return *artifact;
    }
    throw std::invalid_argument("Artifact '" + name + "' not found.");
  }

  virtual std::vector<std::string> listArtifactNames() const { return {}; }

 protected:
  virtual std::optional<Artifact> getArtifactImpl(
      const std::string& name) const {
    (void)name;
    return std::nullopt;
  }

 private:
  friend class cereal::access;

  template <typename Archive>
  void serialize(Archive& archive) {
    (void)archive;
  }
};

using DatasetLoaderFactoryPtr = std::unique_ptr<DatasetLoaderFactory>;

class DatasetLoaderFactoryConfig {
 public:
  virtual DatasetLoaderFactoryPtr createDatasetState(
      const UserInputMap& user_specified_parameters) const = 0;

  virtual ~DatasetLoaderFactoryConfig() = default;

 protected:
  // Private constructor for cereal.
  DatasetLoaderFactoryConfig() {}

 private:
  friend class cereal::access;
  template <class Archive>
  void serialize(Archive& archive) {
    (void)archive;
  }
};

using DatasetLoaderFactoryConfigPtr =
    std::shared_ptr<DatasetLoaderFactoryConfig>;

}  // namespace thirdai::automl::deployment