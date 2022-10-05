#pragma once

#include <cereal/access.hpp>
#include <cereal/types/memory.hpp>
#include <bolt/src/graph/Graph.h>
#include <bolt/src/graph/callbacks/Callback.h>
#include <bolt_vector/src/BoltVector.h>
#include <auto_ml/src/deployment_config/DatasetConfig.h>
#include <auto_ml/src/deployment_config/DeploymentConfig.h>
#include <auto_ml/src/deployment_config/HyperParameter.h>
#include <auto_ml/src/deployment_config/TrainEvalParameters.h>
#include <dataset/src/DataLoader.h>
#include <exceptions/src/Exceptions.h>
#include <limits>
#include <memory>
#include <utility>

namespace thirdai::automl::deployment {

const uint32_t DEFAULT_EVALUATE_BATCH_SIZE = 2048;

/**
 * This class represents an end-to-end data processing + model pipeline. It
 * handles all functionality from loading data to training, evaulation, and
 * inference. The DeploymentConfig acts as a meta config, which specifies
 * what parameters to use, and how to combine them with the user specified
 * parameters to construct the model and dataset processing system.
 */
class ModelPipeline {
 public:
  ModelPipeline(DatasetLoaderFactoryPtr dataset_factory,
                bolt::BoltGraphPtr model,
                TrainEvalParameters train_eval_parameters)
      : _dataset_factory(std::move(dataset_factory)),
        _model(std::move(model)),
        _train_eval_config(train_eval_parameters) {}

  static auto make(const DeploymentConfigPtr& config,
                   const std::unordered_map<std::string, UserParameterInput>&
                       user_specified_parameters) {
    auto [dataset_factory, model] =
        config->createDataLoaderAndModel(user_specified_parameters);

    return ModelPipeline(std::move(dataset_factory), std::move(model),
                         config->train_eval_parameters());
  }

  void trainOnFile(const std::string& filename,
                   const bolt::TrainConfig& train_config, uint32_t batch_size,
                   std::optional<uint32_t> max_in_memory_batches) {
    trainOnDataLoader(
        std::make_shared<dataset::SimpleFileDataLoader>(filename, batch_size),
        train_config, max_in_memory_batches);
  }

  void trainOnDataLoader(
      const std::shared_ptr<dataset::DataLoader>& data_source,
      const bolt::TrainConfig& train_config,
      std::optional<uint32_t> max_in_memory_batches) {
    _dataset_factory->preprocessDataset(data_source, max_in_memory_batches);
    data_source->restart();

    auto dataset = _dataset_factory->getLabeledDatasetLoader(
        data_source, /* training= */ true);

    if (max_in_memory_batches) {
      trainOnStream(dataset, train_config, max_in_memory_batches.value());
    } else {
      trainInMemory(dataset, train_config);
    }
  }

  bolt::InferenceOutputTracker evaulate(const std::string& filename,
                                        bolt::PredictConfig& predict_config) {
    return evaluate(std::make_shared<dataset::SimpleFileDataLoader>(
                        filename, DEFAULT_EVALUATE_BATCH_SIZE),
                    predict_config);
  }

  bolt::InferenceOutputTracker evaluate(
      const std::shared_ptr<dataset::DataLoader>& data_source,
      bolt::PredictConfig& predict_config) {
    auto dataset = _dataset_factory->getLabeledDatasetLoader(
        data_source, /* training= */ false);

    auto [data, labels] =
        dataset->loadInMemory(std::numeric_limits<uint32_t>::max()).value();

    predict_config.returnActivations();

    auto [_, output] = _model->predict({data}, labels, predict_config);

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

    return output;
  }

  BoltVector predict(const std::string& sample, bool use_sparse_inference) {
    std::vector<BoltVector> inputs = _dataset_factory->featurizeInput(sample);

    BoltVector output =
        _model->predictSingle(std::move(inputs), use_sparse_inference);

    if (auto threshold = _train_eval_config.predictionThreshold()) {
      uint32_t prediction_index = argmax(output.activations, output.len);
      if (output.activations[prediction_index] < threshold.value()) {
        output.activations[prediction_index] = threshold.value() + 0.0001;
      }
    }

    return output;
  }

  BoltBatch predictBatch(const std::vector<std::string>& samples,
                         bool use_sparse_inference) {
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

    return outputs;
  }

  void save(const std::string& filename) {
    std::ofstream filestream =
        dataset::SafeFileIO::ofstream(filename, std::ios::binary);
    cereal::BinaryOutputArchive oarchive(filestream);
    oarchive(*this);
  }

  static std::unique_ptr<ModelPipeline> load(const std::string& filename) {
    std::ifstream filestream =
        dataset::SafeFileIO::ifstream(filename, std::ios::binary);
    cereal::BinaryInputArchive iarchive(filestream);
    std::unique_ptr<ModelPipeline> deserialize_into(new ModelPipeline());
    iarchive(*deserialize_into);

    return deserialize_into;
  }

  std::pair<InputDatasets, LabelDataset> loadValidationDataFromFile(
      const std::string& filename) {
    auto file_loader = std::make_shared<dataset::SimpleFileDataLoader>(
        filename, DEFAULT_EVALUATE_BATCH_SIZE);

    return loadValidationDataFromDataLoader(file_loader);
  }

  std::pair<InputDatasets, LabelDataset> loadValidationDataFromDataLoader(
      std::shared_ptr<dataset::DataLoader> data_source) {
    auto loader =
        _dataset_factory->getLabeledDatasetLoader(std::move(data_source),
                                                  /* training= */ false);
    return loader->loadInMemory(std::numeric_limits<uint32_t>::max()).value();
  }

 private:
  // We take in the TrainConfig by value to copy it so we can modify the number
  // epochs.
  void trainInMemory(DatasetLoaderPtr& dataset,
                     bolt::TrainConfig train_config) {
    auto [train_data, train_labels] =
        dataset->loadInMemory(std::numeric_limits<uint32_t>::max()).value();

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
  void trainOnStream(DatasetLoaderPtr& dataset, bolt::TrainConfig train_config,
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

  void trainSingleEpochOnStream(DatasetLoaderPtr& dataset,
                                const bolt::TrainConfig& train_config,
                                uint32_t max_in_memory_batches) {
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

  // Private constructor for cereal.
  ModelPipeline() : _train_eval_config({}, {}, {}, {}) {}

  friend class cereal::access;
  template <class Archive>
  void serialize(Archive& archive) {
    archive(_dataset_factory, _model, _train_eval_config);
  }

  DatasetLoaderFactoryPtr _dataset_factory;
  bolt::BoltGraphPtr _model;
  TrainEvalParameters _train_eval_config;
};

}  // namespace thirdai::automl::deployment