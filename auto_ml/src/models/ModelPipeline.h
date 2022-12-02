#pragma once

#include <cereal/access.hpp>
#include <cereal/types/memory.hpp>
#include "OutputProcessor.h"
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
                bolt::BoltGraphPtr model, OutputProcessorPtr output_processor,
                deployment::TrainEvalParameters train_eval_parameters)
      : _dataset_factory(std::move(dataset_factory)),
        _model(std::move(model)),
        _output_processor(std::move(output_processor)),
        _train_eval_config(train_eval_parameters) {}

  static auto make(
      const deployment::DeploymentConfigPtr& config,
      const std::unordered_map<std::string, deployment::UserParameterInput>&
          user_specified_parameters) {
    auto [dataset_factory, model] =
        config->createDataLoaderAndModel(user_specified_parameters);
    return ModelPipeline(
        std::move(dataset_factory), std::move(model),
        CategoricalOutputProcessor::make(
            config->train_eval_parameters().predictionThreshold()),
        config->train_eval_parameters());
  }

  /**
   * Wrapper around trainOnDataLoader for passing in a filename and batchsize.
   */
  void trainOnFile(const std::string& filename, bolt::TrainConfig& train_config,
                   std::optional<uint32_t> batch_size_opt,
                   const std::optional<ValidationOptions>& validation,
                   std::optional<uint32_t> max_in_memory_batches);

  /**
   * Trains the model on the data given in datasource using the specified
   * TrainConfig and reports any metrics specified in the ValidationOptions on
   * the validation data (if provided). The parameter max_in_memory_batches
   * controls if the data will be processed by streaming chunks of with
   * max_in_memory_batches batches. Note that validation data cannot be used for
   * streaming because of requirements for the order in which data must be
   * loaded with temporal tracking in UDT. See comment in trainOnStream for more
   * details.
   */
  void trainOnDataLoader(
      const std::shared_ptr<dataset::DataLoader>& data_source,
      bolt::TrainConfig& train_config,
      const std::optional<ValidationOptions>& validation,
      std::optional<uint32_t> max_in_memory_batches);

  /**
   * Wrapper around evaluateOnDataLoader for passing in a filename.
   */
  py::object evaluateOnFile(const std::string& filename,
                            std::optional<bolt::EvalConfig>& eval_config_opt);

  /**
   * Processes the data specified in data_source and returns the activations of
   * the final layer. Computes any metrics specifed in the EvalConfig.
   */
  py::object evaluateOnDataLoader(
      const std::shared_ptr<dataset::DataLoader>& data_source,
      std::optional<bolt::EvalConfig>& eval_config_opt);

  /**
   * Takes in a single input sample and returns the activations for the output
   * layer.
   */
  template <typename InputType>
  py::object predict(const InputType& sample, bool use_sparse_inference);

  /**
   * Takes in a batch of input samples and processes them in parallel and
   * returns the activations for the output layer. The order in which the input
   * samples are provided is the order in which the activations are returned.
   */
  template <typename InputBatchType>
  py::object predictBatch(const InputBatchType& samples,
                          bool use_sparse_inference);

  /**
   * Creates an explanation for the prediction of a sample. If the target class
   * is provided then it will instead explain why that class was/was not
   * predicted.
   */
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
  /**
   * Performs in memory training on the given dataset.
   */
  void trainInMemory(data::DatasetLoaderPtr& dataset,
                     bolt::TrainConfig train_config,
                     const std::optional<ValidationOptions>& validation);

  /**
   * Performs training on a streaming dataset in chunks. Note that validation is
   * not used in this case because the validation data must be loaded after the
   * training data if temporal tracking is used in UDT but it is not simple to
   * load validation data after training data for a streaming dataset.
   */
  void trainOnStream(data::DatasetLoaderPtr& dataset,
                     bolt::TrainConfig train_config,
                     uint32_t max_in_memory_batches);

  /**
   * Helper for processing a streaming dataset in chunks for a single epoch.
   */
  void trainSingleEpochOnStream(data::DatasetLoaderPtr& dataset,
                                const bolt::TrainConfig& train_config,
                                uint32_t max_in_memory_batches);

  /**
   * Updates the hash table rebuilding and hash function reconstructing
   * parameters in the TrainConfig if an override for these values is present in
   * the TrainEvalParameters. These parameters cannot be specified by the user
   * for this model, this allows us to override the autotuning of these
   * parameters by specifing them in the TrainEvalParameters in the
   * DeploymentConfig.
   */
  void updateRehashRebuildInTrainConfig(bolt::TrainConfig& train_config);

  friend class cereal::access;
  template <class Archive>
  void serialize(Archive& archive) {
    archive(_dataset_factory, _model, _output_processor, _train_eval_config);
  }

 protected:
  data::DatasetLoaderFactoryPtr _dataset_factory;
  bolt::BoltGraphPtr _model;
  OutputProcessorPtr _output_processor;
  deployment::TrainEvalParameters _train_eval_config;
};

}  // namespace thirdai::automl::models