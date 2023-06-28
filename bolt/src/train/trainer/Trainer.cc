#include "Trainer.h"
#include <bolt/src/nn/ops/FullyConnected.h>
#include <bolt/src/train/metrics/Metric.h>
#include <bolt/src/train/trainer/Dataset.h>
#include <bolt/src/utils/ProgressBar.h>
#include <bolt/src/utils/Timer.h>
#include <dataset/src/Datasets.h>
#include <utils/Logging.h>
#include <chrono>
#include <limits>
#include <optional>
#include <stdexcept>

namespace thirdai::bolt::train {

constexpr uint32_t DEFAULT_BATCH_SIZE = 2048;

Trainer::Trainer(nn::model::ModelPtr model,
                 std::optional<uint32_t> freeze_hash_tables_epoch,
                 InterruptCheck interrupt_check)
    : _model(std::move(model)),
      _epoch(0),
      _freeze_hash_tables_epoch(freeze_hash_tables_epoch),
      _interrupt_check(std::move(interrupt_check)) {
  _history = std::make_shared<metrics::History>();
}

void Trainer::trainOnBatches(
    const LabeledDataset& train_data, const TrainStatePtr& train_state, metrics::MetricCollection& train_metrics,
    callbacks::CallbackList& callbacks, uint32_t *steps_since_validation,
    const std::optional<LabeledDataset>& validation_data, const metrics::InputMetrics& validation_metrics,
    const std::optional<uint32_t>& steps_per_validation, bool use_sparsity_in_validation,
    const std::optional<uint32_t>& logging_interval, bool verbose
) {
    uint32_t num_batches = train_data.first.size();
    auto bar = ProgressBar::makeOptional(verbose, "train", num_batches);

    utils::Timer max_batch_timer;
    for (uint32_t batch_idx = 0; batch_idx < num_batches; batch_idx++) {
        callbacks.onBatchBegin();

        const nn::tensor::TensorList& inputs = train_data.first.at(batch_idx);
        const nn::tensor::TensorList& labels = train_data.second.at(batch_idx);

        _model->trainOnBatch(inputs, labels);
        _model->updateParameters(train_state->learningRate());

        train_metrics.recordBatch(inputs.at(0)->batchSize());

        callbacks.onBatchEnd();

        if (bar) {
            bar->increment();
        }

        *steps_since_validation += *steps_since_validation + 1;
        if (steps_per_validation && *steps_since_validation == *steps_per_validation) {
            validate(*validation_data, validation_metrics, use_sparsity_in_validation);
            *steps_since_validation = 0;
        }

        if (logging_interval && (_model->trainSteps() % *logging_interval) == 0) {
            logging::info(formatIntermediateLogLine(train_metrics.summarizeLastStep()));
        }

        if (train_state->isTrainingStopped()) {
            // TODO: Print stuff and have more graceful termination
            return;
        }

        checkInterrupt();
    }
    train_metrics.updateHistory(*_history);
    std::string log_line = formatTrainLogLine(train_metrics.summarizeLastStep(), num_batches, max_batch_timer.seconds());
    logging::info(log_line);
    if (bar) {
      bar->close(log_line);
    }
}

metrics::History Trainer::train_max_in_memory_batches(
  const dataset::DatasetLoaderPtr& train_data_loader, float learning_rate, uint32_t epochs,
  const metrics::InputMetrics& train_metrics_in, uint32_t batch_size,
  const std::vector<callbacks::CallbackPtr>& callbacks_in,
  std::optional<uint32_t> max_in_memory_batches,
  const std::optional<LabeledDataset>& validation_data,
  const metrics::InputMetrics& validation_metrics,
  std::optional<uint32_t> steps_per_validation,
  bool use_sparsity_in_validation,
  bool autotune_rehash_rebuild, bool verbose, 
  std::optional<uint32_t> logging_interval
){
  auto train_state = TrainState::make(learning_rate);

  metrics::MetricCollection train_metrics(train_metrics_in);

  callbacks::CallbackList callbacks(callbacks_in, _model, train_state,
                                    _history);

  callbacks.onTrainBegin();

  uint32_t steps_since_validation = 0;
  for(_epoch = 0; _epoch < epochs; _epoch++){

    if (_freeze_hash_tables_epoch && _epoch == *_freeze_hash_tables_epoch) {
      _model->freezeHashTables(/* insert_labels_if_not_found= */ true);
    }
    callbacks.onEpochBegin();
    utils::Timer epoch_timer;
    while (auto train_data_wrapper =
                loadSomeWrapper(train_data_loader, batch_size,
                                *max_in_memory_batches, verbose)) {
        auto train_data = train_data_wrapper.value();
        verifyNumBatchesMatch(train_data);
        if (validation_data) {
          verifyNumBatchesMatch(*validation_data);
        }
        if (autotune_rehash_rebuild) {
          autotuneRehashRebuild(train_data.first.size(),
                                train_data.first.at(0).at(0)->batchSize());
        }
        trainOnBatches(train_data, train_state, train_metrics, callbacks, &steps_since_validation,
                          validation_data, validation_metrics, steps_per_validation, use_sparsity_in_validation,
                          logging_interval, verbose);
        }
        epoch_timer.stop();
        (*_history)["epoch_times"].push_back(epoch_timer.seconds());

        train_metrics.reset();

        // This condition ensures that if steps_per_validation coincides with the
        // end of the epoch that we don't validate twice: once above when we reach
        // the validation interval and once when we reach the end of the epoch.
        if (validation_data && steps_since_validation != 0) {
          validate(*validation_data, validation_metrics,
                  use_sparsity_in_validation);
          steps_since_validation = 0;
        }

        callbacks.onEpochEnd();

        if (train_state->isTrainingStopped()) {
          // TODO(Nicholas): Print stuff and have more graceful termination
          return *_history;
        }  

        train_data_loader->restart();
      }
    callbacks.onTrainEnd();

  return *_history;  // Copies the history in case users modify it.
  }

metrics::History Trainer::train(
    const LabeledDataset& train_data, float learning_rate, uint32_t epochs,
    const metrics::InputMetrics& train_metrics_in,
    const std::optional<LabeledDataset>& validation_data,
    const metrics::InputMetrics& validation_metrics,
    std::optional<uint32_t> steps_per_validation,
    bool use_sparsity_in_validation,
    const std::vector<callbacks::CallbackPtr>& callbacks_in,
    bool autotune_rehash_rebuild, bool verbose,
    std::optional<uint32_t> logging_interval
){
    
    verifyNumBatchesMatch(train_data);
    if (validation_data) {
      verifyNumBatchesMatch(*validation_data);
    }
    if (autotune_rehash_rebuild) {
      autotuneRehashRebuild(train_data.first.size(),
                            train_data.first.at(0).at(0)->batchSize());
    }
  auto train_state = TrainState::make(learning_rate);

  metrics::MetricCollection train_metrics(train_metrics_in);

  callbacks::CallbackList callbacks(callbacks_in, _model, train_state,
                                    _history);

  callbacks.onTrainBegin();

  uint32_t steps_since_validation = 0;

  for(_epoch = 0; _epoch < epochs; _epoch++){

    if (_freeze_hash_tables_epoch && _epoch == *_freeze_hash_tables_epoch) {
      _model->freezeHashTables(/* insert_labels_if_not_found= */ true);
    }
    callbacks.onEpochBegin();
    utils::Timer epoch_timer;

    trainOnBatches(train_data, train_state, train_metrics, callbacks, &steps_since_validation,
                       validation_data, validation_metrics, steps_per_validation, use_sparsity_in_validation,
                       logging_interval, verbose);
    epoch_timer.stop();

    (*_history)["epoch_times"].push_back(epoch_timer.seconds());

    train_metrics.reset();

    // This condition ensures that if steps_per_validation coincides with the
    // end of the epoch that we don't validate twice: once above when we reach
    // the validation interval and once when we reach the end of the epoch.
    if (validation_data && steps_since_validation != 0) {
      validate(*validation_data, validation_metrics,
              use_sparsity_in_validation);
      steps_since_validation = 0;
    }

    callbacks.onEpochEnd();

    if (train_state->isTrainingStopped()) {
      // TODO(Nicholas): Print stuff and have more graceful termination
      return *_history;
    }  
  }

  callbacks.onTrainEnd();

  return *_history;  // Copies the history in case users modify it.
}

metrics::History Trainer::train_with_metric_names(
    const LabeledDataset& train_data, float learning_rate, uint32_t epochs,
    const std::vector<std::string>& train_metrics,
    const std::optional<LabeledDataset>& validation_data,
    const std::vector<std::string>& validation_metrics,
    std::optional<uint32_t> steps_per_validation,
    bool use_sparsity_in_validation,
    const std::vector<callbacks::CallbackPtr>& callbacks,
    bool autotune_rehash_rebuild, bool verbose,
    std::optional<uint32_t> logging_interval) {
  return train(
      /* train_data= */ train_data,
      /* learning_rate= */ learning_rate, /* epochs= */ epochs,
      /* train_metrics= */
      metrics::fromMetricNames(_model, train_metrics, "train_"),
      /* validation_data= */ validation_data, /* validation_metrics= */
      metrics::fromMetricNames(_model, validation_metrics, "val_"),
      /* steps_per_validation= */ steps_per_validation,
      /* use_sparsity_in_validation= */ use_sparsity_in_validation,
      /* callbacks= */ callbacks,
      /* autotune_rehash_rebuild= */ autotune_rehash_rebuild,
      /* verbose= */ verbose, /* logging_interval= */ logging_interval);
}

metrics::History Trainer::train_with_dataset_loader(
    const dataset::DatasetLoaderPtr& train_data_loader, float learning_rate,
    uint32_t epochs, uint32_t batch_size,
    std::optional<uint32_t> max_in_memory_batches,
    const metrics::InputMetrics& train_metrics,
    const dataset::DatasetLoaderPtr& validation_data_loader,
    const metrics::InputMetrics& validation_metrics,
    std::optional<uint32_t> steps_per_validation,
    bool use_sparsity_in_validation,
    const std::vector<callbacks::CallbackPtr>& callbacks,
    bool autotune_rehash_rebuild, bool verbose,
    std::optional<uint32_t> logging_interval) {
  if (!max_in_memory_batches) {
    auto train_data = loadAllWrapper(train_data_loader, batch_size, verbose);

    std::optional<LabeledDataset> validation_data = std::nullopt;
    if (validation_data_loader) {
      validation_data =
          loadAllWrapper(validation_data_loader, batch_size, verbose);
    }

    return train(train_data, learning_rate, epochs, train_metrics,
                 validation_data, validation_metrics, steps_per_validation,
                 use_sparsity_in_validation, callbacks, autotune_rehash_rebuild,
                 verbose, logging_interval);
  }

  // We have duplicate code here for loading validation data because for
  // Temporal transformations loading the validation data after the training
  // data is important. We do not do this for the streaming case because it
  // would require doing a first pass over the training data before loading the
  // validation data.
  std::optional<LabeledDataset> validation_data = std::nullopt;
  if (validation_data_loader) {
    validation_data =
        loadAllWrapper(validation_data_loader, batch_size, verbose);
  }

  return train_max_in_memory_batches(
    train_data_loader, learning_rate, epochs, train_metrics, batch_size, 
    callbacks, max_in_memory_batches, validation_data, validation_metrics, 
    steps_per_validation, use_sparsity_in_validation, autotune_rehash_rebuild, 
    verbose, logging_interval);
}

metrics::History Trainer::validate(const LabeledDataset& data,
                                   const metrics::InputMetrics& metrics_in,
                                   bool use_sparsity, bool verbose) {
  metrics::MetricCollection validation_metrics(metrics_in);

  uint32_t num_batches = data.first.size();
  auto bar = ProgressBar::makeOptional(verbose, "validate", num_batches);

  utils::Timer val_timer;

  for (uint32_t batch_idx = 0; batch_idx < num_batches; batch_idx++) {
    const nn::tensor::TensorList& inputs = data.first.at(batch_idx);
    const nn::tensor::TensorList& labels = data.second.at(batch_idx);

    _model->forward(inputs, labels, /* use_sparsity= */ use_sparsity);

    validation_metrics.recordBatch(inputs.at(0)->batchSize());

    if (bar) {
      bar->increment();
    }

    checkInterrupt();
  }

  val_timer.stop();

  validation_metrics.updateHistory(*_history);

  (*_history)["val_times"].push_back(val_timer.seconds());

  std::string log_line = formatValidateLogLine(
      validation_metrics.summarizeLastStep(), num_batches, val_timer.seconds());
  logging::info(log_line);

  if (bar) {
    bar->close(log_line);
  }

  validation_metrics.reset();

  return *_history;  // Copies the history in case users modify it.
}

metrics::History Trainer::validate_with_metric_names(
    const LabeledDataset& data, const std::vector<std::string>& metrics,
    bool use_sparsity, bool verbose) {
  return validate(
      /* data= */ data,
      /* metrics= */ metrics::fromMetricNames(_model, metrics, "val_"),
      /* use_sparsity= */ use_sparsity, /* verbose= */ verbose);
}

metrics::History Trainer::validate_with_dataset_loader(
    const dataset::DatasetLoaderPtr& data, const metrics::InputMetrics& metrics,
    bool use_sparsity, bool verbose) {
  return validate(
      /* data= */ loadAllWrapper(data, /* batch_size= */ DEFAULT_BATCH_SIZE,
                                 verbose),
      /* metrics= */ metrics, /* use_sparsity= */ use_sparsity,
      /* verbose= */ verbose);
}

void Trainer::verifyNumBatchesMatch(const LabeledDataset& data) {
  if (data.first.size() != data.second.size()) {
    throw std::invalid_argument(
        "Data and labels must have same number of batches.");
  }
}

std::string Trainer::formatTrainLogLine(const std::string& metric_summary,
                                        uint32_t batches, int64_t time) {
  std::string logline = fmt::format(
      "train | epoch {} | train_steps {} | {} | train_batches {} | time {}s",
      _epoch, _model->trainSteps(), metric_summary, batches, time);

  return logline;
}

std::string Trainer::formatIntermediateLogLine(
    const std::string& metric_summary) {
  std::string logline =
      fmt::format("train | epoch {} | train_steps {} | {}", _epoch,
                  _model->trainSteps(), metric_summary);

  return logline;
}

std::string Trainer::formatValidateLogLine(const std::string& metric_summary,
                                           uint32_t batches, int64_t time) {
  std::string logline = fmt::format(
      "validate | epoch {} | train_steps {} | {} | val_batches {} | time {}s",
      _epoch, _model->trainSteps(), metric_summary, batches, time);

  return logline;
}

void Trainer::autotuneRehashRebuild(uint32_t num_batches, uint32_t batch_size) {
  for (const auto& op : _model->ops()) {
    if (auto fc = std::dynamic_pointer_cast<nn::ops::FullyConnected>(op)) {
      fc->autotuneRehashRebuild(/* num_batches= */ num_batches,
                                /* batch_size= */ batch_size);
    }
  }
}

LabeledDataset Trainer::loadAllWrapper(
    const dataset::DatasetLoaderPtr& dataset_loader, uint32_t batch_size,
    bool verbose) {
  auto data = loadSomeWrapper(dataset_loader, batch_size,
                              std::numeric_limits<uint32_t>::max(), verbose);
  if (!data) {
    throw std::runtime_error("Unable to load data from data source.");
  }
  return std::move(data.value());
}

std::optional<LabeledDataset> Trainer::loadSomeWrapper(
    const dataset::DatasetLoaderPtr& dataset_loader, uint32_t batch_size,
    uint32_t max_batches, bool verbose) {
  auto datasets = dataset_loader->loadSome(batch_size, max_batches, verbose);
  if (!datasets) {
    return std::nullopt;
  }

  auto input_dims = _model->inputDims();
  auto label_dims = _model->labelDims();

  if (datasets->size() != (input_dims.size() + label_dims.size())) {
    std::stringstream error;
    error << "DatasetLoader generated " << datasets->size()
          << " but the model was expecting " << input_dims.size()
          << " inputs and " << label_dims.size() << " labels.";
    throw std::invalid_argument(error.str());
  }

  std::vector<dataset::BoltDatasetPtr> input_datasets(
      datasets->begin(), datasets->begin() + input_dims.size());

  std::vector<dataset::BoltDatasetPtr> label_datasets(
      datasets->begin() + input_dims.size(), datasets->end());

  return std::make_optional<LabeledDataset>(
      convertDatasets(input_datasets, input_dims, /* copy= */ false),
      convertDatasets(label_datasets, label_dims, /* copy= */ false));
}

}  // namespace thirdai::bolt::train