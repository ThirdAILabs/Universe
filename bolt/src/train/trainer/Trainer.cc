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
#include <utility>

namespace thirdai::bolt {

constexpr uint32_t DEFAULT_BATCH_SIZE = 2048;

Trainer::Trainer(ModelPtr model,
                 std::optional<uint32_t> freeze_hash_tables_epoch,
                 InterruptCheck interrupt_check)
    : _model(std::move(model)),
      _epoch(0),
      _freeze_hash_tables_epoch(freeze_hash_tables_epoch),
      _interrupt_check(std::move(interrupt_check)) {
  _history = std::make_shared<metrics::History>();
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
    std::optional<uint32_t> logging_interval, const DistributedCommPtr& comm) {
  verifyNumBatchesMatch(train_data);
  if (validation_data) {
    verifyNumBatchesMatch(*validation_data);
  }

  if (autotune_rehash_rebuild) {
    autotuneRehashRebuild(train_data.first.size(),
                          train_data.first.at(0).at(0)->batchSize());
  }

  auto train_state = TrainState::make(learning_rate, train_data.first.size());

  metrics::MetricCollection train_metrics(train_metrics_in);

  callbacks::CallbackList callbacks(callbacks_in, _model, train_state,
                                    _history);

  callbacks.onTrainBegin();

  uint32_t steps_since_validation = 0;

  uint32_t num_epochs = _epoch + epochs;
  for (; _epoch < num_epochs; _epoch++) {
    if (_freeze_hash_tables_epoch && _epoch == *_freeze_hash_tables_epoch) {
      _model->freezeHashTables(/* insert_labels_if_not_found= */ true);
    }

    callbacks.onEpochBegin();

    uint32_t num_batches = train_data.first.size();
    if (comm) {
      num_batches = comm->minNumBatches(num_batches);
    }
    auto bar = ProgressBar::makeOptional(verbose, "train", num_batches);

    utils::Timer epoch_timer;

    for (uint32_t batch_idx = 0; batch_idx < num_batches; batch_idx++) {
      callbacks.onBatchBegin();

      const TensorList& inputs = train_data.first.at(batch_idx);
      const TensorList& labels = train_data.second.at(batch_idx);

      utils::Timer train_on_batch_timer;
      _model->trainOnBatch(inputs, labels);

      train_on_batch_timer.stop();
      std::string train_on_batch_log = formatFuncCallLogLine(
          "train_on_batch", batch_idx, train_on_batch_timer.milliseconds());
      logging::info(train_on_batch_log);

      if (comm) {
        comm->communicate(_model);
      }

      utils::Timer update_param_timer;
      _model->updateParameters(train_state->learningRate());

      update_param_timer.stop();

      std::string update_parameter_log = formatFuncCallLogLine(
          "update_parameter", batch_idx, update_param_timer.milliseconds());
      logging::info(update_parameter_log);

      train_metrics.recordBatch(inputs.at(0)->batchSize());

      callbacks.onBatchEnd();

      if (bar) {
        bar->increment();
      }

      ++steps_since_validation;
      if (steps_per_validation &&
          steps_since_validation == *steps_per_validation) {
        validate(*validation_data, validation_metrics,
                 use_sparsity_in_validation);
        steps_since_validation = 0;
      }

      if (logging_interval && (_model->trainSteps() % *logging_interval) == 0) {
        logging::info(
            formatIntermediateLogLine(train_metrics.summarizeLastStep()));
      }

      if (train_state->isTrainingStopped()) {
        // TODO(Nicholas): Print stuff and have more graceful termination
        return *_history;
      }
      checkInterrupt();
    }

    epoch_timer.stop();

    std::vector<std::pair<std::string, float>> metrics_at_rank_0;
    if (comm && train_metrics.hasMetrics()) {
      metrics_at_rank_0 =
          comm->broadcastMetrics(train_metrics.getFlattenedMetrics());
    }

    train_metrics.updateHistory(*_history);

    if (comm && train_metrics.hasMetrics()) {
      train_metrics.setFlattenedMetrics(*_history, metrics_at_rank_0);
    }

    (*_history)["epoch_times"].push_back(epoch_timer.seconds());

    std::string log_line = formatTrainLogLine(
        train_metrics.summarizeLastStep(), num_batches, epoch_timer.seconds());
    logging::info(log_line);

    if (bar) {
      bar->close(log_line);
    }

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
    std::optional<uint32_t> logging_interval, const DistributedCommPtr& comm) {
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
      /* verbose= */ verbose, /* logging_interval= */ logging_interval, comm);
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
    std::optional<uint32_t> logging_interval, const DistributedCommPtr& comm) {
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
                 verbose, logging_interval, comm);
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

  for (uint32_t e = 0; e < epochs; e++) {
    while (auto train_chunk =
               loadSomeWrapper(train_data_loader, batch_size,
                               *max_in_memory_batches, verbose)) {
      train(train_chunk.value(), learning_rate, /* epochs= */ 1, train_metrics,
            validation_data, validation_metrics, steps_per_validation,
            use_sparsity_in_validation, callbacks, autotune_rehash_rebuild,
            verbose, logging_interval, comm);
    }
    train_data_loader->restart();
  }

  return *_history;
}

metrics::History Trainer::train_with_data_loader(
    const data::LoaderPtr& train_data_loader, float learning_rate,
    uint32_t epochs, std::optional<size_t> max_in_memory_batches,
    const metrics::InputMetrics& train_metrics,
    const data::LoaderPtr& validation_data_loader,
    const metrics::InputMetrics& validation_metrics,
    std::optional<uint32_t> steps_per_validation,
    bool use_sparsity_in_validation,
    const std::vector<callbacks::CallbackPtr>& callbacks,
    bool autotune_rehash_rebuild, bool verbose,
    std::optional<uint32_t> logging_interval, const DistributedCommPtr& comm) {
  if (!max_in_memory_batches) {
    auto train_data = train_data_loader->all();

    std::optional<LabeledDataset> validation_data = std::nullopt;
    if (validation_data_loader) {
      validation_data = validation_data_loader->all();
    }

    return train(train_data, learning_rate, epochs, train_metrics,
                 validation_data, validation_metrics, steps_per_validation,
                 use_sparsity_in_validation, callbacks, autotune_rehash_rebuild,
                 verbose, logging_interval, comm);
  }

  // We have duplicate code here for loading validation data because for
  // Temporal transformations loading the validation data after the training
  // data is important. We do not do this for the streaming case because it
  // would require doing a first pass over the training data before loading the
  // validation data.
  std::optional<LabeledDataset> validation_data = std::nullopt;
  if (validation_data_loader) {
    validation_data = validation_data_loader->all();
  }

  for (uint32_t e = 0; e < epochs; e++) {
    while (auto train_chunk = train_data_loader->next(*max_in_memory_batches)) {
      train(train_chunk.value(), learning_rate, /* epochs= */ 1, train_metrics,
            validation_data, validation_metrics, steps_per_validation,
            use_sparsity_in_validation, callbacks, autotune_rehash_rebuild,
            verbose, logging_interval, comm);
    }
    train_data_loader->restart();
  }

  return *_history;
}

metrics::History Trainer::validate(const LabeledDataset& data,
                                   const metrics::InputMetrics& metrics_in,
                                   bool use_sparsity, bool verbose) {
  metrics::MetricCollection validation_metrics(metrics_in);

  uint32_t num_batches = data.first.size();
  auto bar = ProgressBar::makeOptional(verbose, "validate", num_batches);

  utils::Timer val_timer;

  for (uint32_t batch_idx = 0; batch_idx < num_batches; batch_idx++) {
    const TensorList& inputs = data.first.at(batch_idx);
    const TensorList& labels = data.second.at(batch_idx);

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

metrics::History Trainer::validate_with_data_loader(
    const data::LoaderPtr& data, const metrics::InputMetrics& metrics,
    bool use_sparsity, bool verbose) {
  return validate(/* data= */ data->all(), /* metrics= */ metrics,
                  /* use_sparsity= */ use_sparsity, /* verbose= */ verbose);
}

void Trainer::verifyNumBatchesMatch(const LabeledDataset& data) {
  if (data.first.size() != data.second.size()) {
    throw std::invalid_argument(
        "Data and labels must have same number of batches.");
  }
}

std::string Trainer::formatTrainLogLine(const std::string& metric_summary,
                                        uint32_t batches, double time) {
  std::string logline = fmt::format(
      "train | epoch {} | train_steps {} | {} | train_batches {} | time "
      "{:.3f}s",
      _epoch, _model->trainSteps(), metric_summary, batches, time);

  return logline;
}

std::string Trainer::formatFuncCallLogLine(const std::string& func_call,
                                           uint32_t batches, int64_t time) {
  std::string logline = fmt::format(
      "func {} | epoch {} | train_steps {} | train_batches {} | time {} ms",
      func_call, _epoch, _model->trainSteps(), batches, time);

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
                                           uint32_t batches, double time) {
  std::string logline = fmt::format(
      "validate | epoch {} | train_steps {} | {} | val_batches {} | time "
      "{:.3f}s",
      _epoch, _model->trainSteps(), metric_summary, batches, time);

  return logline;
}

void Trainer::autotuneRehashRebuild(uint32_t num_batches, uint32_t batch_size) {
  for (const auto& op : _model->ops()) {
    if (auto fc = FCKernelOp::cast(op)) {
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

}  // namespace thirdai::bolt