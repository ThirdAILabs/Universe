#pragma once

#include <cereal/access.hpp>
#include <bolt/src/graph/InferenceOutputTracker.h>
#include <bolt/src/graph/nodes/Input.h>
#include <bolt_vector/src/BoltVector.h>
#include <auto_ml/src/Aliases.h>
#include <dataset/src/DataSource.h>
#include <dataset/src/Datasets.h>
#include <dataset/src/StreamingGenericDatasetLoader.h>
#include <dataset/src/batch_processors/GenericBatchProcessor.h>
#include <dataset/src/blocks/BlockInterface.h>
#include <dataset/src/dataset_loaders/DatasetLoader.h>
#include <cstdint>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <variant>

namespace thirdai::automl::data {

/**
 * Structure of Dataset Loading:
 * DatasetLoaderFactory:
 *      takes in DataSources, for instance S3, a file, etc. and returns a
 *      DatasetLoader for the given resource. This factory can also maintain any
 *      state that's needed to load datasets, for instance in the Tabular and
 *      Sequential data sources which are stateful.
 *
 * DatasetLoader:
 *      Can return bolt datasets form the associated data source until it is
 *      exhausted. For instance a data source would be returned by the factory
 *      for each data source train or evaluate is invoked with.
 *
 */

class DatasetLoaderFactory {
 public:
  virtual dataset::DatasetLoaderPtr getLabeledDatasetLoader(
      std::shared_ptr<dataset::DataSource> data_source, bool training) = 0;

  virtual std::vector<BoltVector> featurizeInput(const LineInput& input) = 0;

  virtual std::vector<BoltVector> featurizeInput(const MapInput& input) {
    (void)input;
    throw std::invalid_argument(
        "This model pipeline configuration does not support map input. Pass in "
        "a string instead.");
  };

  virtual std::vector<BoltBatch> featurizeInputBatch(
      const LineInputBatch& inputs) = 0;

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

  virtual std::string className(uint32_t neuron_id) const {
    (void)neuron_id;
    throw std::runtime_error(
        "This model cannot map ids to string labels since it assumes integer "
        "labels; the ids and labels are equivalent.");
  }

  virtual bool hasTemporalTracking() const = 0;

  virtual ~DatasetLoaderFactory() = default;

 private:
  friend class cereal::access;

  template <typename Archive>
  void serialize(Archive& archive) {
    (void)archive;
  }
};

using DatasetLoaderFactoryPtr = std::shared_ptr<DatasetLoaderFactory>;

}  // namespace thirdai::automl::data