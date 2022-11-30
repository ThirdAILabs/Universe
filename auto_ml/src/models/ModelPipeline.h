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
#include <telemetry/src/PrometheusClient.h>
#include <limits>
#include <memory>
#include <string>
#include <unordered_map>
#include <utility>

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
                   std::optional<uint32_t> max_in_memory_batches) {
    uint32_t batch_size =
        batch_size_opt.value_or(_train_eval_config.defaultBatchSize());
    trainOnDataLoader(dataset::SimpleFileDataLoader::make(filename, batch_size),
                      train_config, validation, max_in_memory_batches);
  }

  void trainOnDataLoader(
      const std::shared_ptr<dataset::DataLoader>& data_source,
      bolt::TrainConfig& train_config,
      const std::optional<ValidationOptions>& validation,
      std::optional<uint32_t> max_in_memory_batches) {
    auto start_time = std::chrono::system_clock::now();

    _dataset_factory->preprocessDataset(data_source, max_in_memory_batches);
    data_source->restart();

    auto dataset = _dataset_factory->getLabeledDatasetLoader(
        data_source, /* training= */ true);

    updateRehashRebuildInTrainConfig(train_config);

    if (max_in_memory_batches) {
      trainOnStream(dataset, train_config, max_in_memory_batches.value());
    } else {
      trainInMemory(dataset, train_config, validation);
    }

    std::chrono::duration<double> elapsed_time =
        std::chrono::system_clock::now() - start_time;
    telemetry::client.trackTraining(
        /* training_time_seconds = */ elapsed_time.count());
  }

  bolt::InferenceOutputTracker evaulate(
      const std::string& filename,
      std::optional<bolt::EvalConfig>& eval_config_opt) {
    return evaluate(dataset::SimpleFileDataLoader::make(
                        filename, DEFAULT_EVALUATE_BATCH_SIZE),
                    eval_config_opt);
  }

  bolt::InferenceOutputTracker evaluate(
      const std::shared_ptr<dataset::DataLoader>& data_source,
      std::optional<bolt::EvalConfig>& eval_config_opt) {
    auto start_time = std::chrono::system_clock::now();

    auto dataset = _dataset_factory->getLabeledDatasetLoader(
        data_source, /* training= */ false);

    auto [data, labels] =
        dataset->loadInMemory(std::numeric_limits<uint32_t>::max()).value();

    bolt::EvalConfig eval_config =
        eval_config_opt.value_or(bolt::EvalConfig::makeConfig());

    eval_config.returnActivations();

    auto [_, output] = _model->evaluate({data}, labels, eval_config);

    if (auto threshold = _train_eval_config.predictionThreshold()) {
      uint32_t output_dim = output.numNonzerosInOutput();
      for (uint32_t i = 0; i < output.numSamples(); i++) {
        float* activations = output.activationsForSample(i);
        uint32_t prediction_index = argmax(activations, output_dim);

        if (activations[prediction_index] < threshold.value()) {
          activations[prediction_index] = threshold.value() + 0.0001;
        }
      }
    }

    auto evaluate_output = _dataset_factory->processEvaluateOutput(output);

    std::chrono::duration<double> elapsed_time =
        std::chrono::system_clock::now() - start_time;
    telemetry::client.trackEvaluate(
        /* evaluate_time_seconds = */ elapsed_time.count());

    return evaluate_output;
  }

  template <typename InputType>
  BoltVector predict(const InputType& sample, bool use_sparse_inference) {
    auto start_time = std::chrono::system_clock::now();

    std::vector<BoltVector> inputs = _dataset_factory->featurizeInput(sample);

    BoltVector output =
        _model->predictSingle(std::move(inputs), use_sparse_inference);

    if (auto threshold = _train_eval_config.predictionThreshold()) {
      uint32_t prediction_index = argmax(output.activations, output.len);
      if (output.activations[prediction_index] < threshold.value()) {
        output.activations[prediction_index] = threshold.value() + 0.0001;
      }
    }

    auto prediction = _dataset_factory->processOutputVector(output);

    std::chrono::duration<double> elapsed_time =
        std::chrono::system_clock::now() - start_time;
    telemetry::client.trackPredictions(
        /* inference_time_seconds = */ elapsed_time.count(),
        /* num_inferences = */ 1);

    return prediction;
  }

  template <typename InputBatchType>
  BoltBatch predictBatch(const InputBatchType& samples,
                         bool use_sparse_inference) {
    auto start_time = std::chrono::system_clock::now();

    std::vector<BoltBatch> input_batches =
        _dataset_factory->featurizeInputBatch(samples);

    BoltBatch outputs = _model->predictSingleBatch(std::move(input_batches),
                                                   use_sparse_inference);

    if (auto threshold = _train_eval_config.predictionThreshold()) {
      for (auto& output : outputs) {
        uint32_t prediction_index = argmax(output.activations, output.len);
        if (output.activations[prediction_index] < threshold.value()) {
          output.activations[prediction_index] = threshold.value() + 0.0001;
        }
      }
    }

    for (auto& vector : outputs) {
      vector = _dataset_factory->processOutputVector(vector);
    }

    std::chrono::duration<double> elapsed_time =
        std::chrono::system_clock::now() - start_time;
    telemetry::client.trackPredictions(
        /* inference_time_seconds = */ elapsed_time.count(),
        /* num_inferences = */ outputs.getBatchSize());

    return outputs;
  }

  template <typename InputType>
  std::vector<dataset::Explanation> explain(
      const InputType& sample,
      std::optional<std::variant<uint32_t, std::string>> target_class =
          std::nullopt) {
    auto start_time = std::chrono::system_clock::now();

    std::optional<uint32_t> target_neuron;
    if (target_class) {
      target_neuron = _dataset_factory->labelToNeuronId(*target_class);
    }

    auto [gradients_indices, gradients_ratio] = _model->getInputGradientSingle(
        /* input_data= */ {_dataset_factory->featurizeInput(sample)},
        /* explain_prediction_using_highest_activation= */ true,
        /* neuron_to_explain= */ target_neuron);
    auto explanation =
        _dataset_factory->explain(gradients_indices, gradients_ratio, sample);

    std::chrono::duration<double> elapsed_time =
        std::chrono::system_clock::now() - start_time;
    telemetry::client.trackExplanation(
        /* explain_time_seconds = */ elapsed_time.count());

    return explanation;
  }

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
                     const std::optional<ValidationOptions>& validation) {
    auto [train_data, train_labels] =
        dataset->loadInMemory(std::numeric_limits<uint32_t>::max()).value();

    if (validation) {
      auto validation_dataset = _dataset_factory->getLabeledDatasetLoader(
          dataset::SimpleFileDataLoader::make(validation->filename(),
                                              DEFAULT_EVALUATE_BATCH_SIZE),
          /* training= */ false);

      auto [val_data, val_labels] =
          validation_dataset->loadInMemory(std::numeric_limits<uint32_t>::max())
              .value();

      train_config.withValidation(val_data, val_labels,
                                  validation->validationConfig());
    }

    uint32_t epochs = train_config.epochs();

    if (_train_eval_config.freezeHashTables() && epochs > 1) {
      train_config.setEpochs(/* new_epochs=*/1);

      _model->train(train_data, train_labels, train_config);

      _model->freezeHashTables(/* insert_labels_if_not_found= */ true);

      train_config.setEpochs(/* new_epochs= */ epochs - 1);
    }

    _model->train(train_data, train_labels, train_config);
  }

  // We take in the TrainConfig by value to copy it so we can modify the number
  // epochs.
  void trainOnStream(data::DatasetLoaderPtr& dataset,
                     bolt::TrainConfig train_config,
                     uint32_t max_in_memory_batches) {
    uint32_t epochs = train_config.epochs();
    // We want a single epoch in the train config in order to train for a single
    // epoch for each pass over the dataset.
    train_config.setEpochs(/* new_epochs= */ 1);

    if (_train_eval_config.freezeHashTables() && epochs > 1) {
      trainSingleEpochOnStream(dataset, train_config, max_in_memory_batches);
      _model->freezeHashTables(/* insert_labels_if_not_found= */ true);

      --epochs;
    }

    for (uint32_t e = 0; e < epochs; e++) {
      trainSingleEpochOnStream(dataset, train_config, max_in_memory_batches);
    }
  }

  void trainSingleEpochOnStream(data::DatasetLoaderPtr& dataset,
                                const bolt::TrainConfig& train_config,
                                uint32_t max_in_memory_batches) {
    while (auto datasets = dataset->loadInMemory(max_in_memory_batches)) {
      auto& [data, labels] = datasets.value();

      _model->train({data}, labels, train_config);
    }

    dataset->restart();
  }

  void updateRehashRebuildInTrainConfig(bolt::TrainConfig& train_config) {
    if (auto hash_table_rebuild =
            _train_eval_config.rebuildHashTablesInterval()) {
      train_config.withRebuildHashTables(hash_table_rebuild.value());
    }

    if (auto reconstruct_hash_fn =
            _train_eval_config.reconstructHashFunctionsInterval()) {
      train_config.withReconstructHashFunctions(reconstruct_hash_fn.value());
    }
  }

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

}  // namespace thirdai::automl::models