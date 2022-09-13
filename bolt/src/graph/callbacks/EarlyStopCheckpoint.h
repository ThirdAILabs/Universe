#pragma once

#include "Callback.h"
#include <bolt/src/graph/ExecutionConfig.h>
#include <bolt/src/graph/Graph.h>
#include <dataset/src/Datasets.h>
#include <functional>
#include <limits>

namespace thirdai::bolt {

/**
 * @brief This callback is intended to stop training early based on prediction
 * results from a given validation set. Saves the best model to model_save_path
 *
 * @param monitored_metric The metric to monitor for early stopping. Should be a
 * valid metric name with an additional prefix of either 'train_' or 'val_' to
 * associate it to training or validation data respectively.
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
 *
 * TODO(david): Validation data should ideally be moved to the train level and
 * this callback should only monitor changes in validation metrics. Let's
 * refactor this when the validation data needs to be used elsewhere.
 */
class EarlyStopCheckpoint : public Callback {
 public:
  EarlyStopCheckpoint(std::string monitored_metric, std::string model_save_path,
                      uint32_t patience = 2, double min_delta = 0)
      : _monitored_metric(monitored_metric),
        _model_save_path(std::move(model_save_path)),
        _patience(patience),
        _min_delta(std::abs(min_delta)) {
    std::cout << _monitored_metric << std::endl;
    initValidationTrackers();
  }

  void onTrainBegin(BoltGraph& model, TrainState& train_state) final {
    (void)model;
    (void)train_state;
    initValidationTrackers();
  }

  void onEpochEnd(BoltGraph& model, TrainState& train_state) final {
    double metric_val = train_state.getMetricValue(_monitored_metric);

    if (isImprovement(metric_val)) {
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
  bool isImprovement(double metric_val) const {
    if (_should_minimize) {
      return metric_val + _min_delta < _best_validation_score;
    }
    return metric_val - _min_delta > _best_validation_score;
  }

  void initValidationTrackers() {
    std::string real_metric_name = stripMetricPrefixes(_monitored_metric);
    _epochs_since_best = 0;
    _should_minimize = makeMetric(real_metric_name)->smallerIsBetter();

    _best_validation_score = _should_minimize
                                 ? std::numeric_limits<double>::max()
                                 : std::numeric_limits<double>::min();
  }

  static std::string stripMetricPrefixes(std::string prefixed_metric_name) {
    std::vector<std::string> available_prefixes = {"train_", "val_"};
    for (const auto& prefix : available_prefixes) {
      if (prefix == prefixed_metric_name.substr(0, prefix.size())) {
        return prefixed_metric_name.substr(
            prefix.size(), prefixed_metric_name.size() - prefix.size());
      }
    }
    throw std::invalid_argument(
        "Metric is not prefixed correctly. Metrics should be prefixed with "
        "'train_' or 'val_' to correctly distinguish between training and "
        "validation data. ");
  }

  std::string _monitored_metric;
  std::string _model_save_path;
  uint32_t _patience;
  double _min_delta;

  // Below are variables used to track the best validation score over the course
  // of a train call. These are reset in onTrainBegin(..) so they can be reused.
  uint32_t _epochs_since_best;
  bool _should_minimize;
  double _best_validation_score;
};

using EarlyStopCheckpointPtr = std::shared_ptr<EarlyStopCheckpoint>;

}  // namespace thirdai::bolt