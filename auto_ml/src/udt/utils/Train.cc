#include "Train.h"
#include <auto_ml/src/udt/Defaults.h>
#include <dataset/src/Datasets.h>

namespace thirdai::automl::udt::utils {

namespace {

/**
 * Copies all metrics from the "from" map to the "to" map, extending the end of
 * the "to" map. Modifes in place.
 */
void aggregateMetrics(bolt::MetricData& to, const bolt::MetricData& from) {
  for (const auto& [metric_name, metric_values] : from) {
    if (!to.count(metric_name)) {
      to[metric_name] = std::vector<double>();
    }
    for (const double val : metric_values) {
      to[metric_name].push_back(val);
    }
  }
}

bolt::MetricData trainSingleEpochOnStream(
    bolt::BoltGraphPtr& model, dataset::DatasetLoaderPtr& dataset_loader,
    const bolt::TrainConfig& train_config, uint32_t max_in_memory_batches,
    size_t batch_size, licensing::TrainPermissionsToken token) {
  bolt::MetricData aggregated_metrics;
  while (auto datasets = dataset_loader->loadSome(
             batch_size, /* num_batches = */ max_in_memory_batches,
             /* verbose = */ train_config.verbose())) {
    auto [data, labels] = splitDataLabels(std::move(datasets.value()));

    auto partial_metrics = model->train({data}, labels, train_config, token);
    aggregateMetrics(/* to = */ aggregated_metrics,
                     /* from = */ partial_metrics);
  }

  dataset_loader->restart();

  return aggregated_metrics;
}

bolt::MetricData trainOnStream(bolt::BoltGraphPtr& model,
                               dataset::DatasetLoaderPtr& dataset_loader,
                               bolt::TrainConfig train_config,
                               size_t batch_size, size_t max_in_memory_batches,
                               bool freeze_hash_tables,
                               licensing::TrainPermissionsToken token) {
  uint32_t epochs = train_config.epochs();
  // We want a single epoch in the train config in order to train for a single
  // epoch for each pass over the dataset.
  train_config.setEpochs(/* new_epochs= */ 1);

  bolt::MetricData aggregated_metrics;

  if (freeze_hash_tables && epochs > 1) {
    auto partial_metrics =
        trainSingleEpochOnStream(model, dataset_loader, train_config,
                                 max_in_memory_batches, batch_size, token);
    aggregateMetrics(/* to = */ aggregated_metrics,
                     /* from = */ partial_metrics);

    model->freezeHashTables(/* insert_labels_if_not_found= */ true);

    --epochs;
  }

  for (uint32_t e = 0; e < epochs; e++) {
    auto partial_metrics =
        trainSingleEpochOnStream(model, dataset_loader, train_config,
                                 max_in_memory_batches, batch_size, token);

    aggregateMetrics(/* to = */ aggregated_metrics,
                     /* from = */ partial_metrics);
  }

  return aggregated_metrics;
}

bolt::MetricData trainInMemory(bolt::BoltGraphPtr& model,
                               dataset::DatasetLoaderPtr& dataset_loader,
                               bolt::TrainConfig train_config,
                               size_t batch_size, bool freeze_hash_tables,
                               licensing::TrainPermissionsToken token) {
  auto loaded_data = dataset_loader->loadAll(
      /* batch_size = */ batch_size, /* verbose = */ train_config.verbose());
  auto [train_data, train_labels] = splitDataLabels(std::move(loaded_data));

  uint32_t epochs = train_config.epochs();

  if (freeze_hash_tables && epochs > 1) {
    train_config.setEpochs(/* new_epochs=*/1);

    model->train(train_data, train_labels, train_config, token);

    model->freezeHashTables(/* insert_labels_if_not_found= */ true);

    train_config.setEpochs(/* new_epochs= */ epochs - 1);
  }

  return model->train(train_data, train_labels, train_config, token);
}

}  // namespace

bolt::MetricData train(bolt::BoltGraphPtr& model,
                       dataset::DatasetLoaderPtr& dataset_loader,
                       const bolt::TrainConfig& train_config, size_t batch_size,
                       std::optional<size_t> max_in_memory_batches,
                       bool freeze_hash_tables,
                       licensing::TrainPermissionsToken token) {
  if (max_in_memory_batches) {
    return trainOnStream(model, dataset_loader, train_config, batch_size,
                         max_in_memory_batches.value(), freeze_hash_tables,
                         token);
  }
  return trainInMemory(model, dataset_loader, train_config, batch_size,
                       freeze_hash_tables, token);
}

bolt::TrainConfig getTrainConfig(
    uint32_t epochs, float learning_rate,
    const std::optional<ValidationDatasetLoader>& validation,
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
  if (validation) {
    auto val_dataset = validation->first->loadAll(
        /* batch_size= */ defaults::BATCH_SIZE, verbose);
    auto [val_data, val_labels] = splitDataLabels(std::move(val_dataset));

    bolt::EvalConfig val_config =
        getEvalConfig(validation->second.metrics(),
                      validation->second.sparseInference(), verbose);

    train_config.withValidation(
        val_data, val_labels, val_config,
        /* validation_frequency = */
        validation->second.stepsPerValidation().value_or(0));
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

// Splits a vector of datasets as returned by a dataset loader (where the labels
// are the last dataset in the list)
std::pair<dataset::BoltDatasetList, dataset::BoltDatasetPtr> splitDataLabels(
    dataset::BoltDatasetList&& datasets) {
  auto labels = datasets.back();
  datasets.pop_back();
  return {datasets, labels};
}

}  // namespace thirdai::automl::udt::utils