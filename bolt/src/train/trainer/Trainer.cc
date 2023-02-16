#include "Trainer.h"
#include <bolt/src/utils/ProgressBar.h>
#include <bolt/src/utils/Timer.h>
#include <utils/Logging.h>
#include <chrono>

namespace thirdai::bolt::train {

Trainer::Trainer(nn::model::ModelPtr model)
    : _model(std::move(model)), _epoch(0) {
  _history = std::make_shared<metrics::History>();
}

metrics::History Trainer::train(
    const LabeledDataset& train_data, uint32_t epochs, float learning_rate,
    const metrics::InputMetrics& train_metrics_in,
    const std::optional<LabeledDataset>& validation_data,
    const metrics::InputMetrics& validation_metrics_in,
    std::optional<uint32_t> steps_per_validation,
    const std::vector<callbacks::CallbackPtr>& callbacks_in) {
  auto train_state = TrainState::make(learning_rate);

  metrics::MetricCollection train_metrics(train_metrics_in);
  metrics::MetricCollection validation_metrics(validation_metrics_in);

  callbacks::CallbackList callbacks(callbacks_in, _model, train_state,
                                    _history);
  callbacks.onTrainBegin();

  for (uint32_t epoch = 0; epoch < epochs; epoch++) {
    trainEpoch(train_data, train_state, train_metrics, validation_data,
               validation_metrics, steps_per_validation, callbacks);
    _epoch++;
  }

  callbacks.onTrainEnd();

  return *_history;
}

metrics::History Trainer::trainStream(
    const dataset::DatasetLoaderPtr& train_data, size_t batch_size,
    std::optional<size_t> max_in_memory_batches, uint32_t epochs,
    float learning_rate, const metrics::InputMetrics& train_metrics_in,
    const std::optional<LabeledDataset>& validation_data,
    const metrics::InputMetrics& validation_metrics_in,
    std::optional<uint32_t> steps_per_validation,
    const std::vector<callbacks::CallbackPtr>& callbacks_in) {
  if (!max_in_memory_batches) {
    return train(train_data->loadAllTensor(batch_size), epochs, learning_rate,
                 train_metrics_in, validation_data, validation_metrics_in,
                 steps_per_validation, callbacks_in);
  }

  auto train_state = TrainState::make(learning_rate);

  metrics::MetricCollection train_metrics(train_metrics_in);
  metrics::MetricCollection validation_metrics(validation_metrics_in);

  callbacks::CallbackList callbacks(callbacks_in, _model, train_state,
                                    _history);

  callbacks.onTrainBegin();

  for (uint32_t epoch = 0; epoch < epochs; epoch++) {
    while (auto data =
               train_data->loadSomeTensor(batch_size, *max_in_memory_batches)) {
      trainEpoch(*data, train_state, train_metrics, validation_data,
                 validation_metrics, steps_per_validation, callbacks);
      if (train_state->isTrainingStopped()) {
        return *_history;
      }
    }
    _epoch++;
    train_data->restart();
  }

  callbacks.onTrainEnd();

  return *_history;
}

void Trainer::trainEpoch(const LabeledDataset& train_data,
                         const TrainStatePtr& train_state,
                         metrics::MetricCollection& train_metrics,
                         const std::optional<LabeledDataset>& validation_data,
                         metrics::MetricCollection& validation_metrics,
                         std::optional<uint32_t> steps_per_validation,
                         callbacks::CallbackList& callbacks) {
  verifyNumBatchesMatch(train_data);
  if (validation_data) {
    verifyNumBatchesMatch(*validation_data);
  }

  callbacks.onEpochBegin();

  uint32_t num_batches = train_data.first.size();
  ProgressBar bar("train", num_batches);

  utils::Timer epoch_timer;

  for (uint32_t batch_idx = 0; batch_idx < num_batches; batch_idx++) {
    callbacks.onBatchBegin();

    _model->trainOnBatch(train_data.first.at(batch_idx),
                         train_data.second.at(batch_idx));

    _model->updateParameters(train_state->learningRate());

    train_metrics.recordBatch(
        train_data.first.at(batch_idx).at(0)->batchSize());

    callbacks.onBatchEnd();

    bar.increment();

    train_state->incrementStepsSinceValidation();
    if (steps_per_validation && validation_data &&
        train_state->stepsSinceValidation() == *steps_per_validation) {
      validate(*validation_data, validation_metrics);
      train_state->resetStepsSinceValidation();
    }

    if (train_state->isTrainingStopped()) {
      // TODO(Nicholas): Print stuff and have more graceful termination
      return;
    }
  }

  epoch_timer.stop();

  train_metrics.updateHistory(_history, /*prefix= */ "train_");

  (*_history)["epoch_times"].push_back(epoch_timer.seconds());

  std::string log_line = formatTrainLogLine(train_metrics.summarizeLastStep(),
                                            num_batches, epoch_timer.seconds());
  bar.close(log_line);
  logging::info(log_line);

  train_metrics.reset();

  callbacks.onEpochEnd();

  // This condition ensures that if we steps_per_validation coincides with the
  // end of the epoch that we don't validate twice: once above when we reach
  // the validation interval and once when we reach the end of the epoch.
  if (validation_data && train_state->stepsSinceValidation() != 0) {
    validate(*validation_data, validation_metrics);
    train_state->resetStepsSinceValidation();
  }
}

void Trainer::validate(const LabeledDataset& validation_data,
                       metrics::MetricCollection& validation_metrics) {
  uint32_t num_batches = validation_data.first.size();
  ProgressBar bar("validate", num_batches);

  utils::Timer val_timer;

  for (uint32_t batch_idx = 0; batch_idx < num_batches; batch_idx++) {
    // TODO(Nicholas): Add option to use sparsity for validation.
    _model->forward(validation_data.first.at(batch_idx),
                    validation_data.second.at(batch_idx),
                    /* use_sparsity= */ false);

    validation_metrics.recordBatch(
        validation_data.first.at(batch_idx).at(0)->batchSize());

    bar.increment();
  }

  val_timer.stop();

  validation_metrics.updateHistory(_history, /* prefix= */ "val_");

  (*_history)["val_times"].push_back(val_timer.seconds());

  std::string log_line = formatValidateLogLine(
      validation_metrics.summarizeLastStep(), num_batches, val_timer.seconds());
  bar.close(log_line);
  logging::info(log_line);

  validation_metrics.reset();
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

std::string Trainer::formatValidateLogLine(const std::string& metric_summary,
                                           uint32_t batches, int64_t time) {
  std::string logline = fmt::format(
      "validate | epoch {} | train_steps {} | {} | val_batches {} | time {}s",
      _epoch, _model->trainSteps(), metric_summary, batches, time);

  return logline;
}

}  // namespace thirdai::bolt::train