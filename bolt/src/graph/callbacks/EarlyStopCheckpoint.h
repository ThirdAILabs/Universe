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
 * @param predict_config configurations for evaluation on the given validation
 * data. must include metrics
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
  EarlyStopCheckpoint(std::vector<dataset::BoltDatasetPtr> validation_data,
                      dataset::BoltDatasetPtr validation_labels,
                      PredictConfig predict_config, std::string model_save_path,
                      uint32_t patience = 2, double min_delta = 0)
      : _validation_data(std::move(validation_data)),
        _validation_labels(std::move(validation_labels)),
        _predict_config(std::move(predict_config)),
        _model_save_path(std::move(model_save_path)),
        _patience(patience),
        _min_delta(std::abs(min_delta)) {
    uint32_t num_metrics = _predict_config.getMetricNames().size();
    if (num_metrics != 1) {
      throw std::invalid_argument(
          "Validation-based early stopping only supports the use of one "
          "metric, passed in " +
          std::to_string(num_metrics) + " metrics.");
    }
  }

  void onTrainBegin(BoltGraph& model) final {
    (void)model;

    std::string metric_name = _predict_config.getMetricNames()[0];

    // setting these onTrainBegin allows callback instances to be reused
    _epochs_since_best = 0;
    _should_stop_training = false;
    _should_minimize = makeMetric(metric_name)->smallerIsBetter();
    _best_validation_score = _should_minimize
                                 ? std::numeric_limits<double>::max()
                                 : std::numeric_limits<double>::min();
  }

  void onEpochEnd(BoltGraph& model) final {
    std::string metric_name = _predict_config.getMetricNames()[0];
    double metric_val =
        model.predict(_validation_data, _validation_labels, _predict_config)
            .first[metric_name];

    _epochs_since_best++;
    if (isImprovement(metric_val)) {
      _best_validation_score = metric_val;
      _epochs_since_best = 0;
      model.save(_model_save_path);
    } else if (_epochs_since_best == _patience) {
      _should_stop_training = true;
    }
  }

  bool shouldStopTraining() final { return _should_stop_training; }

 private:
  bool isImprovement(double metric_val) const {
    if (_should_minimize) {
      return metric_val + _min_delta < _best_validation_score;
    }
    return metric_val - _min_delta > _best_validation_score;
  }

  std::vector<dataset::BoltDatasetPtr> _validation_data;
  dataset::BoltDatasetPtr _validation_labels;
  PredictConfig _predict_config;
  std::string _model_save_path;
  uint32_t _patience;
  double _min_delta;

  bool _should_stop_training;
  uint32_t _epochs_since_best;
  bool _should_minimize;
  double _best_validation_score;
};

using EarlyStopCheckpointPtr = std::shared_ptr<EarlyStopCheckpoint>;

}  // namespace thirdai::bolt