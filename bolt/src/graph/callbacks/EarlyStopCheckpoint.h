#pragma once

#include "Callback.h"
#include <bolt/src/graph/ExecutionConfig.h>
#include <bolt/src/graph/Graph.h>
#include <bolt/src/metrics/Metric.h>
#include <dataset/src/Datasets.h>
#include <functional>
#include <limits>

namespace thirdai::bolt {

/**
 * @brief This callback is intended to stop training early based on prediction
 * results from a given validation set. Saves the best model to model_save_path
 *
 * @param monitored_metric The metric to monitor for early stopping. The metric
 * is assumed to be associated with validation data.
 * @param model_save_path file path to save the model that scored the
 * best on the validation set
 * @param patience number of epochs with no improvement in validation score
 * after which training will be stopped.
 * @param min_delta minimum change in the monitored quantity to qualify as an
 * improvement, i.e. an absolute change of less than min_delta, will count as no
 * improvement.
 *
 * Based on the keras design found here:
 * https://keras.io/api/callbacks/early_stopping/
 */
class EarlyStopCheckpoint : public Callback {
 public:
  EarlyStopCheckpoint(std::string monitored_metric, std::string model_save_path,
                      uint32_t patience = 2, double min_delta = 0)
      : _metric(makeMetric(monitored_metric)),
        _model_save_path(std::move(model_save_path)),
        _patience(patience),
        _epochs_since_best(0),
        _best_validation_score(_metric->worst()),
        _min_delta(std::abs(min_delta)) {}

  void onTrainBegin(BoltGraph& model, TrainState& train_state) final {
    (void)model;
    (void)train_state;
  }

  void onEpochEnd(BoltGraph& model, TrainState& train_state) final {
    double metric_val =
        train_state.getValidationMetrics(_metric->name()).back();

    if (std::abs(metric_val - _best_validation_score) > _min_delta &&
        _metric->betterThan(metric_val, _best_validation_score)) {
      _best_validation_score = metric_val;
      _epochs_since_best = 0;
      model.save(_model_save_path);
      return;
    }

    _epochs_since_best++;
    if (_epochs_since_best == _patience) {
      train_state.stop_training = true;
    }
  }

 private:
  std::shared_ptr<Metric> _metric;
  std::string _model_save_path;
  uint32_t _patience;

  uint32_t _epochs_since_best;
  double _best_validation_score;
  double _min_delta;
};

using EarlyStopCheckpointPtr = std::shared_ptr<EarlyStopCheckpoint>;

}  // namespace thirdai::bolt