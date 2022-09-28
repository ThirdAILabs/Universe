#pragma once

#include <bolt/src/graph/Graph.h>
#include <bolt_vector/src/BoltVector.h>
#include <auto_ml/src/deployment_config/DatasetConfig.h>
#include <auto_ml/src/deployment_config/DeploymentConfig.h>
#include <auto_ml/src/deployment_config/HyperParameter.h>
#include <auto_ml/src/deployment_config/TrainEvalParameters.h>
#include <dataset/src/DataLoader.h>
#include <limits>
#include <utility>

namespace thirdai::automl {

/**
 * This class represents an end-to-end data processing + model pipeline. It
 * handles all functionality from loading data to training, evaulation, and
 * inference. The DeploymentConfig acts as a meta config, which specifies
 * what parameters to use, and how to combine them with the user specified
 * parameters to construct the model and dataset processing system.
 */
class ModelPipeline {
 public:
  ModelPipeline(const deployment_config::DeploymentConfigPtr& config,
                const std::unordered_map<std::string,
                                         deployment_config::UserParameterInput>&
                    user_specified_parameters)
      : _train_eval_config(config->parameters()) {
    auto [dataset_state, model] =
        config->createDataLoaderAndModel(user_specified_parameters);

    _model = std::move(model);
    _dataset_factory = std::move(dataset_state);
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
    auto dataset = _dataset_factory->getDatasetLoader(data_source);

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
    auto dataset = _dataset_factory->getDatasetLoader(data_source);

    auto [data, labels] =
        dataset->loadInMemory(std::numeric_limits<uint32_t>::max()).value();

    bolt::PredictConfig predict_cfg =
        bolt::PredictConfig::makeConfig()
            .withMetrics(_train_eval_config.evaluationMetrics())
            .returnActivations();
    if (_train_eval_config.useSparseInference()) {
      predict_cfg.enableSparseInference();
    }

    auto output = _model->predict({data}, {labels}, predict_cfg);

    // TODO(Nicholas): add option for thresholding (wayfair use case)
    return output;
  }

  BoltVector predict(const std::string& sample) {
    std::vector<BoltVector> inputs = _dataset_factory->featurizeInput(sample);

    BoltVector output = _model->predictSingle(
        std::move(inputs), _train_eval_config.useSparseInference());

    // TODO(Nicholas): add option for thresholding (wayfair use case)
    return output;
  }

  BoltBatch predictBatch(const std::vector<std::string>& samples) {
    std::vector<BoltBatch> input_batches =
        _dataset_factory->featurizeInputBatch(samples);

    BoltBatch outputs = _model->predictSingleBatch(
        std::move(input_batches), _train_eval_config.useSparseInference());

    return outputs;
  }

  uint32_t defaultBatchSize() const {
    return _train_eval_config.defaultBatchSize();
  }

 private:
  void trainInMemory(deployment_config::DatasetLoaderPtr& dataset,
                     float learning_rate, uint32_t epochs) {
    auto [train_data, train_labels] =
        dataset->loadInMemory(std::numeric_limits<uint32_t>::max()).value();

    if (_train_eval_config.useSparseInference() && epochs > 1) {
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
    if (_train_eval_config.useSparseInference() && epochs > 1) {
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
            _train_eval_config.rebuildHashTablesInterval()) {
      train_config.withRebuildHashTables(hash_table_rebuild.value());
    }

    if (auto reconstruct_hash_fn =
            _train_eval_config.reconstructHashFunctionsInterval()) {
      train_config.withReconstructHashFunctions(reconstruct_hash_fn.value());
    }
    return train_config;
  }

  deployment_config::TrainEvalParameters _train_eval_config;
  bolt::BoltGraphPtr _model;
  deployment_config::DatasetLoaderFactoryPtr _dataset_factory;
};

}  // namespace thirdai::automl