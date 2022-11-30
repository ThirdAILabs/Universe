#pragma once

#include <cereal/access.hpp>
#include <cereal/types/memory.hpp>
#include <bolt/src/graph/Graph.h>
#include <bolt_vector/src/BoltVector.h>
#include <auto_ml/src/deployment_config/DatasetConfig.h>
#include <auto_ml/src/deployment_config/DeploymentConfig.h>
#include <auto_ml/src/deployment_config/HyperParameter.h>
#include <auto_ml/src/deployment_config/TrainEvalParameters.h>
#include <dataset/src/DataLoader.h>
#include <dataset/src/blocks/BlockInterface.h>
#include <exceptions/src/Exceptions.h>
#include <pybind11/pybind11.h>
#include <limits>
#include <memory>
#include <string>
#include <unordered_map>
#include <utility>

namespace py = pybind11;

namespace thirdai::automl::models {

const uint32_t DEFAULT_EVALUATE_BATCH_SIZE = 2048;

class ValidationOptions {
 public:
  ValidationOptions(std::string filename, std::vector<std::string> metrics,
                    std::optional<uint32_t> interval, bool use_sparse_inference)
      : _filename(std::move(filename)),
        _metrics(std::move(metrics)),
        _interval(interval),
        _use_sparse_inference(use_sparse_inference) {}

  const std::string& filename() const { return _filename; }

  // TODO(Nicholas): Refactor ValidationContext to use an optional to indicate
  // validation batch frequency instead of having 0 be a special value.
  uint32_t interval() const { return _interval.value_or(0); }

  bolt::EvalConfig validationConfig() const {
    bolt::EvalConfig val_config =
        bolt::EvalConfig::makeConfig().withMetrics(_metrics);

    if (_use_sparse_inference) {
      val_config.enableSparseInference();
    }

    return val_config;
  }

 private:
  std::string _filename;
  std::vector<std::string> _metrics;
  std::optional<uint32_t> _interval;
  bool _use_sparse_inference;
};

/**
 * This class represents an end-to-end data processing + model pipeline. It
 * handles all functionality from loading data to training, evaulation, and
 * inference. The DeploymentConfig acts as a meta config, which specifies
 * what parameters to use, and how to combine them with the user specified
 * parameters to construct the model and dataset processing system.
 */
class ModelPipeline {
 public:
  ModelPipeline(data::DatasetLoaderFactoryPtr dataset_factory,
                bolt::BoltGraphPtr model,
                deployment::TrainEvalParameters train_eval_parameters)
      : _dataset_factory(std::move(dataset_factory)),
        _model(std::move(model)),
        _train_eval_config(train_eval_parameters) {}

  static auto make(
      const deployment::DeploymentConfigPtr& config,
      const std::unordered_map<std::string, deployment::UserParameterInput>&
          user_specified_parameters) {
    auto [dataset_factory, model] =
        config->createDataLoaderAndModel(user_specified_parameters);
    return ModelPipeline(std::move(dataset_factory), std::move(model),
                         config->train_eval_parameters());
  }

  void trainOnFile(const std::string& filename, bolt::TrainConfig& train_config,
                   std::optional<uint32_t> batch_size_opt,
                   const std::optional<ValidationOptions>& validation,
                   std::optional<uint32_t> max_in_memory_batches);

  void trainOnDataLoader(
      const std::shared_ptr<dataset::DataLoader>& data_source,
      bolt::TrainConfig& train_config,
      const std::optional<ValidationOptions>& validation,
      std::optional<uint32_t> max_in_memory_batches);

  py::object evaluateOnFile(const std::string& filename,
                            std::optional<bolt::EvalConfig>& eval_config_opt);

  py::object evaluateOnDataLoader(
      const std::shared_ptr<dataset::DataLoader>& data_source,
      std::optional<bolt::EvalConfig>& eval_config_opt);

  template <typename InputType>
  py::object predict(const InputType& sample, bool use_sparse_inference);

  template <typename InputBatchType>
  py::object predictBatch(const InputBatchType& samples,
                          bool use_sparse_inference);

  template <typename InputType>
  std::vector<dataset::Explanation> explain(
      const InputType& sample,
      std::optional<std::variant<uint32_t, std::string>> target_class =
          std::nullopt);

  uint32_t defaultBatchSize() const {
    return _train_eval_config.defaultBatchSize();
  }

  void save(const std::string& filename) {
    std::ofstream filestream =
        dataset::SafeFileIO::ofstream(filename, std::ios::binary);
    cereal::BinaryOutputArchive oarchive(filestream);
    oarchive(*this);
  }

  static std::shared_ptr<ModelPipeline> load(const std::string& filename) {
    std::ifstream filestream =
        dataset::SafeFileIO::ifstream(filename, std::ios::binary);
    cereal::BinaryInputArchive iarchive(filestream);
    std::shared_ptr<ModelPipeline> deserialize_into(new ModelPipeline());
    iarchive(*deserialize_into);

    return deserialize_into;
  }

  data::DatasetLoaderFactoryPtr getDataProcessor() const {
    return _dataset_factory;
  }

 protected:
  // Protected constructor for cereal.
  // Protected so derived classes can also use it for serialization purposes.
  ModelPipeline() : _train_eval_config({}, {}, {}, {}, {}) {}

 private:
  // We take in the TrainConfig by value to copy it so we can modify the number
  // epochs.
  void trainInMemory(data::DatasetLoaderPtr& dataset,
                     bolt::TrainConfig train_config,
                     const std::optional<ValidationOptions>& validation);

  // We take in the TrainConfig by value to copy it so we can modify the number
  // epochs.
  void trainOnStream(data::DatasetLoaderPtr& dataset,
                     bolt::TrainConfig train_config,
                     uint32_t max_in_memory_batches);

  void trainSingleEpochOnStream(data::DatasetLoaderPtr& dataset,
                                const bolt::TrainConfig& train_config,
                                uint32_t max_in_memory_batches);

  void updateRehashRebuildInTrainConfig(bolt::TrainConfig& train_config);

  static uint32_t argmax(const float* const array, uint32_t len) {
    assert(len > 0);

    uint32_t max_index = 0;
    float max_value = array[0];
    for (uint32_t i = 1; i < len; i++) {
      if (array[i] > max_value) {
        max_index = i;
        max_value = array[i];
      }
    }
    return max_index;
  }

  friend class cereal::access;
  template <class Archive>
  void serialize(Archive& archive) {
    archive(_dataset_factory, _model, _train_eval_config);
  }

 protected:
  data::DatasetLoaderFactoryPtr _dataset_factory;
  bolt::BoltGraphPtr _model;
  deployment::TrainEvalParameters _train_eval_config;
};

py::object convertInferenceTrackerToNumpy(bolt::InferenceOutputTracker& output);

py::object convertBoltVectorToNumpy(const BoltVector& vector);

py::object convertBoltBatchToNumpy(const BoltBatch& batch);

}  // namespace thirdai::automl::models