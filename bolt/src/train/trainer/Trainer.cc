#include "Trainer.h"
#include <bolt/src/train/metrics/Metric.h>
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
    const LabeledDataset& train_data, float learning_rate, uint32_t epochs,
    const metrics::InputMetrics& train_metrics_in,
    const std::optional<LabeledDataset>& validation_data,
    const metrics::InputMetrics& validation_metrics,
    std::optional<uint32_t> steps_per_validation,
    bool use_sparsity_in_validation,
    const std::vector<callbacks::CallbackPtr>& callbacks_in) {
  verifyNumBatchesMatch(train_data);
  if (validation_data) {
    verifyNumBatchesMatch(*validation_data);
  }

  auto train_state = TrainState::make(learning_rate);

  metrics::MetricCollection train_metrics(train_metrics_in);

  callbacks::CallbackList callbacks(callbacks_in, _model, train_state,
                                    _history);

  callbacks.onTrainBegin();

  uint32_t steps_since_validation = 0;

  uint32_t num_epochs = _epoch + epochs;
  for (; _epoch < num_epochs; _epoch++) {
    callbacks.onEpochBegin();

    uint32_t num_batches = train_data.first.size();
    ProgressBar bar("train", num_batches);

    utils::Timer epoch_timer;

    for (uint32_t batch_idx = 0; batch_idx < num_batches; batch_idx++) {
      callbacks.onBatchBegin();

      _model->trainOnBatch(train_data.first.at(batch_idx),
                           train_data.second.at(batch_idx));

      _model->updateParameters(train_state->learningRate());

      train_metrics.recordBatch(train_data.first.at(batch_idx)->batchSize());

      callbacks.onBatchEnd();

      bar.increment();

      ++steps_since_validation;
      if (steps_per_validation &&
          steps_since_validation == *steps_per_validation) {
        validate(*validation_data, validation_metrics,
                 use_sparsity_in_validation);
        steps_since_validation = 0;
      }

      if (train_state->isTrainingStopped()) {
        // TODO(Nicholas): Print stuff and have more graceful termination
        return *_history;
      }
    }

    epoch_timer.stop();

    train_metrics.updateHistory(_history);

    (*_history)["epoch_times"].push_back(epoch_timer.seconds());

    std::string log_line = formatTrainLogLine(
        train_metrics.summarizeLastStep(), num_batches, epoch_timer.seconds());
    bar.close(log_line);
    logging::info(log_line);

    train_metrics.reset();

    callbacks.onEpochEnd();

    // This condition ensures that if we steps_per_validation coincides with the
    // end of the epoch that we don't validate twice: once above when we reach
    // the validation interval and once when we reach the end of the epoch.
    if (validation_data && steps_since_validation != 0) {
      validate(*validation_data, validation_metrics,
               use_sparsity_in_validation);
      steps_since_validation = 0;
    }
  }

  callbacks.onTrainEnd();

  return *_history;  // Copies the history in case users modify it.
}

metrics::History Trainer::validate(
    const LabeledDataset& validation_data,
    const metrics::InputMetrics& validation_metrics_in, bool use_sparsity) {
  metrics::MetricCollection validation_metrics(validation_metrics_in);

  uint32_t num_batches = validation_data.first.size();
  ProgressBar bar("validate", num_batches);

  utils::Timer val_timer;

  for (uint32_t batch_idx = 0; batch_idx < num_batches; batch_idx++) {
    // TODO(Nicholas): Add option to use sparsity for validation.
    _model->forward(validation_data.first.at(batch_idx),
                    validation_data.second.at(batch_idx),
                    /* use_sparsity= */ use_sparsity);

    validation_metrics.recordBatch(
        validation_data.first.at(batch_idx)->batchSize());

    bar.increment();
  }

  val_timer.stop();

  validation_metrics.updateHistory(_history);

  (*_history)["val_times"].push_back(val_timer.seconds());

  std::string log_line = formatValidateLogLine(
      validation_metrics.summarizeLastStep(), num_batches, val_timer.seconds());
  bar.close(log_line);
  logging::info(log_line);

  validation_metrics.reset();

  return *_history;  // Copies the history in case users modify it.
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