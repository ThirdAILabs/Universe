#pragma once

#include <bolt/src/graph/Graph.h>
#include <bolt_vector/src/BoltVector.h>
#include <auto_ml/src/deployment_config/DatasetConfig.h>
#include <auto_ml/src/deployment_config/DeploymentConfig.h>
#include <auto_ml/src/deployment_config/HyperParameter.h>
#include <dataset/src/DataLoader.h>
#include <limits>
#include <utility>

namespace thirdai::automl {

class ModelPipeline {
 public:
  ModelPipeline(deployment_config::DeploymentConfigPtr config,
                const std::optional<std::string>& option,
                const std::unordered_map<std::string,
                                         deployment_config::UserParameterInput>&
                    user_specified_parameters)
      : _config(std::move(config)) {
    auto [dataset_state, model] =
        _config->createDataLoaderAndModel(option, user_specified_parameters);

    _model = std::move(model);
    _dataset_state = std::move(dataset_state);
  }

  void train(const std::string& filename, uint32_t epochs, float learning_rate,
             std::optional<uint32_t> batch_size_opt,
             std::optional<uint32_t> max_in_memory_batches) {
    uint32_t batch_size = batch_size_opt.value_or(defaultBatchSize());

    train(std::make_shared<dataset::SimpleFileDataLoader>(filename, batch_size),
          epochs, learning_rate, max_in_memory_batches);
  }

  void train(const std::shared_ptr<dataset::DataLoader>& data_source,
             uint32_t epochs, float learning_rate,
             std::optional<uint32_t> max_in_memory_batches) {
    auto dataset = _dataset_state->getDatasetLoader(data_source);

    if (max_in_memory_batches) {
      trainOnStream(dataset, learning_rate, epochs,
                    max_in_memory_batches.value());
    } else {
      trainInMemory(dataset, learning_rate, epochs);
    }
  }

  bolt::InferenceResult evaulate(const std::string& filename) {
    return evaluate(std::make_shared<dataset::SimpleFileDataLoader>(
        filename, defaultBatchSize()));
  }

  bolt::InferenceResult evaluate(
      const std::shared_ptr<dataset::DataLoader>& data_source) {
    auto dataset = _dataset_state->getDatasetLoader(data_source);

    auto [data, labels] =
        dataset->loadInMemory(std::numeric_limits<uint32_t>::max()).value();

    bolt::PredictConfig predict_cfg =
        bolt::PredictConfig::makeConfig()
            .withMetrics(_config->parameters().evaluationMetrics())
            .returnActivations();
    if (_config->parameters().useSparseInference()) {
      predict_cfg.enableSparseInference();
    }

    auto output = _model->predict({data}, {labels}, predict_cfg);

    // TODO(Nicholas): add option for thresholding (wayfair use case)
    return output;
  }

  BoltVector predict(const std::string& sample) {
    std::vector<BoltVector> inputs = _dataset_state->featurizeInput(sample);

    BoltVector output = _model->predictSingle(
        std::move(inputs), _config->parameters().useSparseInference());

    // TODO(Nicholas): add option for thresholding (wayfair use case)
    return output;
  }

  BoltBatch predictBatch(const std::vector<std::string>& samples) {
    std::vector<std::vector<BoltVector>> inputs(samples.size());

#pragma omp parallel for default(none) shared(inputs, samples)
    for (uint32_t i = 0; i < samples.size(); i++) {
      inputs[i] = _dataset_state->featurizeInput(samples[i]);
    }

    // TODO(Nicholas): convert vector of vector of inputs to vector of batches.
    std::vector<BoltBatch> input_batches;

    BoltBatch outputs = _model->predictSingleBatch(
        std::move(input_batches), _config->parameters().useSparseInference());

    return outputs;
  }

  uint32_t defaultBatchSize() const {
    return _config->parameters().defaultBatchSize();
  }

 private:
  void trainInMemory(deployment_config::DatasetLoaderPtr& dataset,
                     float learning_rate, uint32_t epochs) {
    auto [train_data, train_labels] =
        dataset->loadInMemory(std::numeric_limits<uint32_t>::max()).value();

    if (_config->parameters().useSparseInference() && epochs > 1) {
      bolt::TrainConfig train_cfg_initial =
          getTrainConfig(learning_rate, /* epochs= */ 1);

      _model->train(train_data, train_labels, train_cfg_initial);

      _model->freezeHashTables(/* insert_labels_if_not_found= */ true);

      --epochs;
    }

    bolt::TrainConfig train_cfg = getTrainConfig(learning_rate, epochs);
    _model->train(train_data, train_labels, train_cfg);
  }

  void trainOnStream(deployment_config::DatasetLoaderPtr& dataset,
                     float learning_rate, uint32_t epochs,
                     uint32_t max_in_memory_batches) {
    if (_config->parameters().useSparseInference() && epochs > 1) {
      trainSingleEpochOnStream(dataset, learning_rate, max_in_memory_batches);
      _model->freezeHashTables(/* insert_labels_if_not_found= */ true);

      --epochs;
    }

    for (uint32_t e = 0; e < epochs; e++) {
      trainSingleEpochOnStream(dataset, learning_rate, max_in_memory_batches);
    }
  }

  void trainSingleEpochOnStream(deployment_config::DatasetLoaderPtr& dataset,
                                float learning_rate,
                                uint32_t max_in_memory_batches) {
    bolt::TrainConfig train_config =
        getTrainConfig(learning_rate, /* epochs= */ 1);

    while (auto datasets = dataset->loadInMemory(max_in_memory_batches)) {
      auto& [data, labels] = datasets.value();

      _model->train({data}, labels, train_config);
    }

    dataset->restart();
  }

  bolt::TrainConfig getTrainConfig(float learning_rate, uint32_t epochs) {
    bolt::TrainConfig train_config =
        bolt::TrainConfig::makeConfig(learning_rate, epochs);

    if (auto hash_table_rebuild =
            _config->parameters().rebuildHashTablesInterval()) {
      train_config.withRebuildHashTables(hash_table_rebuild.value());
    }

    if (auto reconstruct_hash_fn =
            _config->parameters().reconstructHashFunctionsInterval()) {
      train_config.withReconstructHashFunctions(reconstruct_hash_fn.value());
    }
    return train_config;
  }

  deployment_config::DeploymentConfigPtr _config;
  bolt::BoltGraphPtr _model;
  deployment_config::DatasetStatePtr _dataset_state;
};

}  // namespace thirdai::automl