#include "Train.h"
#include <auto_ml/src/udt/Defaults.h>
#include <auto_ml/src/udt/UDTBackend.h>
#include <dataset/src/DataSource.h>
#include <dataset/src/dataset_loaders/DatasetLoader.h>
#include <functional>

namespace thirdai::automl::udt::utils {

namespace {

void trainSingleEpochOnStream(bolt::BoltGraphPtr& model,
                              dataset::DatasetLoaderPtr& dataset_loader,
                              const bolt::TrainConfig& train_config,
                              uint32_t max_in_memory_batches, size_t batch_size,
                              licensing::TrainPermissionsToken token) {
  while (auto datasets = dataset_loader->loadSome(
             batch_size, /* num_batches = */ max_in_memory_batches,
             /* verbose = */ train_config.verbose())) {
    auto& [data, labels] = datasets.value();

    model->train({data}, labels, train_config, token);
  }

  dataset_loader->restart();
}

void trainOnStream(bolt::BoltGraphPtr& model,
                   dataset::DatasetLoaderPtr& dataset_loader,
                   bolt::TrainConfig train_config, size_t batch_size,
                   size_t max_in_memory_batches, bool freeze_hash_tables,
                   licensing::TrainPermissionsToken token) {
  uint32_t epochs = train_config.epochs();
  // We want a single epoch in the train config in order to train for a single
  // epoch for each pass over the dataset.
  train_config.setEpochs(/* new_epochs= */ 1);

  if (freeze_hash_tables && epochs > 1) {
    trainSingleEpochOnStream(model, dataset_loader, train_config,
                             max_in_memory_batches, batch_size, token);
    model->freezeHashTables(/* insert_labels_if_not_found= */ true);

    --epochs;
  }

  for (uint32_t e = 0; e < epochs; e++) {
    trainSingleEpochOnStream(model, dataset_loader, train_config,
                             max_in_memory_batches, batch_size, token);
  }
}

}  // namespace

void trainInMemory(
    bolt::BoltGraphPtr& model,
    std::pair<dataset::InputDatasets, dataset::LabelDataset> datasets,
    bolt::TrainConfig train_config, bool freeze_hash_tables,
    licensing::TrainPermissionsToken token) {
  auto [train_data, train_labels] = std::move(datasets);

  uint32_t epochs = train_config.epochs();

  if (freeze_hash_tables && epochs > 1) {
    train_config.setEpochs(/* new_epochs=*/1);

    model->train(train_data, train_labels, train_config);

    model->freezeHashTables(/* insert_labels_if_not_found= */ true);

    train_config.setEpochs(/* new_epochs= */ epochs - 1);
  }

  model->train(train_data, train_labels, train_config, token);
}

void train(bolt::BoltGraphPtr& model, dataset::DatasetLoaderPtr& dataset_loader,
           const bolt::TrainConfig& train_config, size_t batch_size,
           std::optional<size_t> max_in_memory_batches, bool freeze_hash_tables,
           licensing::TrainPermissionsToken token) {
  if (max_in_memory_batches) {
    trainOnStream(model, dataset_loader, train_config, batch_size,
                  max_in_memory_batches.value(), freeze_hash_tables, token);
  } else {
    auto loaded_data = dataset_loader->loadAll(
        /* batch_size = */ batch_size, /* verbose = */ train_config.verbose());
    trainInMemory(model, loaded_data, train_config, freeze_hash_tables, token);
  }
}

bolt::TrainConfig getTrainConfig(
    uint32_t epochs, float learning_rate,
    const std::vector<std::string>& train_metrics,
    const std::vector<std::shared_ptr<bolt::Callback>>& callbacks, bool verbose,
    std::optional<uint32_t> logging_interval) {
  bolt::TrainConfig train_config =
      bolt::TrainConfig::makeConfig(learning_rate, epochs)
          .withMetrics(train_metrics)
          .withCallbacks(callbacks);
  if (logging_interval) {
    train_config.withLogLossFrequency(*logging_interval);
  }
  if (!verbose) {
    train_config.silence();
  }
  return train_config;
}

bolt::TrainConfig getTrainConfig(
    uint32_t epochs, float learning_rate,
    const std::optional<Validation>& validation,
    const std::vector<std::string>& train_metrics,
    const std::vector<std::shared_ptr<bolt::Callback>>& callbacks, bool verbose,
    std::optional<uint32_t> logging_interval,
    dataset::DatasetLoaderPtr validation_dataset_loader) {
  bolt::TrainConfig train_config =
      getTrainConfig(epochs, learning_rate, train_metrics, callbacks, verbose,
                     logging_interval);
  if (validation && validation_dataset_loader) {
    auto val_data = validation_dataset_loader->loadAll(
        /* batch_size= */ defaults::BATCH_SIZE, verbose);

    bolt::EvalConfig val_config = getEvalConfig(
        validation->metrics(), validation->sparseInference(), verbose);

    train_config.withValidation(val_data.first, val_data.second, val_config,
                                /* validation_frequency = */
                                validation->stepsPerValidation().value_or(0));
  }

  return train_config;
}

bolt::EvalConfig getEvalConfig(const std::vector<std::string>& metrics,
                               bool sparse_inference, bool verbose,
                               bool validation) {
  bolt::EvalConfig eval_config =
      bolt::EvalConfig::makeConfig().withMetrics(metrics);
  if (sparse_inference) {
    eval_config.enableSparseInference();
  }
  if (!verbose) {
    eval_config.silence();
  }
  if (!validation) {
    eval_config.returnActivations();
  }

  return eval_config;
}

uint32_t predictedClass(const BoltVector& activation_vec,
                        std::optional<float> binary_threshold) {
  if (!binary_threshold.has_value()) {
    return activation_vec.getHighestActivationId();
  }
  return activation_vec.activations[1] >= *binary_threshold;
}

py::object predictedClasses(bolt::InferenceOutputTracker& output,
                            std::optional<float> binary_threshold) {
  utils::NumpyArray<uint32_t> predictions(output.numSamples());
  for (uint32_t i = 0; i < output.numSamples(); i++) {
    BoltVector activation_vec = output.getSampleAsNonOwningBoltVector(i);
    predictions.mutable_at(i) =
        predictedClass(activation_vec, binary_threshold);
  }
  return py::object(std::move(predictions));
}

py::object predictedClasses(const BoltBatch& outputs,
                            std::optional<float> binary_threshold) {
  utils::NumpyArray<uint32_t> predictions(outputs.getBatchSize());
  for (uint32_t i = 0; i < outputs.getBatchSize(); i++) {
    predictions.mutable_at(i) = predictedClass(outputs[i], binary_threshold);
  }
  return py::object(std::move(predictions));
}

}  // namespace thirdai::automl::udt::utils