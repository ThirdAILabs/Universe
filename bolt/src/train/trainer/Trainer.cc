#include "Trainer.h"
#include <bolt/src/utils/ProgressBar.h>
#include <utils/Logging.h>
#include <chrono>

namespace thirdai::bolt::train {

static auto now() { return std::chrono::high_resolution_clock::now(); }

// NOLINTNEXTLINE
#define between(start, end) \
  std::chrono::duration_cast<std::chrono::seconds>((end) - (start)).count()

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
  // TODO(Nicholas): Check datasets have same number of batches and batchsize.

  auto train_state = TrainState::make(learning_rate);

  metrics::MetricList train_metrics(train_metrics_in, _model);
  metrics::MetricList validation_metrics(validation_metrics_in, _model);

  callbacks::CallbackList callbacks(callbacks_in, _model, train_state,
                                    _history);

  callbacks.onTrainBegin();

  uint32_t steps_since_validation = 0;

  uint32_t num_epochs = _epoch + epochs;
  for (; _epoch < num_epochs; _epoch++) {
    callbacks.onEpochBegin();

    uint32_t num_batches = train_data.first->numBatches();
    ProgressBar bar("train", num_batches);

    auto epoch_start = now();

    for (uint32_t batch_idx = 0; batch_idx < num_batches; batch_idx++) {
      callbacks.onBatchBegin();

      _model->trainOnBatchSingleInput(train_data.first->at(batch_idx),
                                      train_data.second->at(batch_idx));

      _model->updateParameters(train_state->learningRate());

      train_metrics.recordBatch(train_data.first->at(batch_idx).getBatchSize());

      callbacks.onBatchEnd();

      bar.increment();

      ++steps_since_validation;
      if (steps_per_validation &&
          steps_since_validation == *steps_per_validation) {
        validate(*validation_data, validation_metrics);
        steps_since_validation = 0;
      }

      if (train_state->isTrainingStopped()) {
        // TODO(Nicholas): Print stuff and have more graceful termination
        return *_history;
      }
    }

    auto epoch_end = now();
    int64_t time = between(epoch_start, epoch_end);

    train_metrics.updateHistory(_history, /*prefix= */ "train_");

    (*_history)["all"]["epoch_times"].push_back(static_cast<double>(time));

    std::string log_line = formatTrainLogLine(train_metrics.summarizeLastStep(),
                                              num_batches, time);
    bar.close(log_line);
    logging::info(log_line);

    train_metrics.reset();

    callbacks.onEpochEnd();

    // This condition ensures that if we validated after the last batch already
    // that we don't do so again here.
    if (validation_data && steps_since_validation != 0) {
      validate(*validation_data, validation_metrics);
      steps_since_validation = 0;
    }
  }

  callbacks.onTrainEnd();

  return *_history;
}

void Trainer::validate(const LabeledDataset& validation_data,
                       metrics::MetricList& validation_metrics) {
  uint32_t num_batches = validation_data.first->numBatches();
  ProgressBar bar("validate", num_batches);

  auto val_start = now();

  for (uint32_t batch_idx = 0; batch_idx < num_batches; batch_idx++) {
    // TODO(Nicholas): Add option to use sparsity for validation.
    _model->forwardSingleInput(validation_data.first->at(batch_idx),
                               /* use_sparsity= */ false);

    _model->setSingleLabel(validation_data.second->at(batch_idx));

    validation_metrics.recordBatch(
        validation_data.first->at(batch_idx).getBatchSize());

    bar.increment();
  }

  auto val_end = now();
  int64_t time = between(val_start, val_end);

  validation_metrics.updateHistory(_history, /* prefix= */ "val_");

  (*_history)["all"]["val_time"].push_back(static_cast<double>(time));

  std::string log_line = formatValidateLogLine(
      validation_metrics.summarizeLastStep(), num_batches, time);
  bar.close(log_line);
  logging::info(log_line);

  validation_metrics.reset();
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