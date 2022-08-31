#pragma once

#include "Callback.h"
#include <bolt/src/graph/ExecutionConfig.h>
#include <bolt/src/graph/Graph.h>
#include <dataset/src/Datasets.h>
#include <functional>
#include <limits>

namespace thirdai::bolt {

/**
 * This callback is intended to stop training early based on prediction results
 * from a given validation set. We give this callback a "patience" argument,
 * which tells us how many extra epochs we'll train for without beating our
 * previous best validation metric.
 *
 * This callback always resets the model's weights at the end of training to use
 * its best checkpoint based on validation performance.
 */
class EarlyStopValidation : public Callback {
 public:
  EarlyStopValidation(dataset::BoltDatasetList validation_data,
                      dataset::BoltDatasetPtr validation_labels,
                      PredictConfig predict_config, uint32_t patience = 2)
      : _validation_data(std::move(validation_data)),
        _validation_labels(std::move(validation_labels)),
        _patience(patience),
        _predict_config(std::move(predict_config)) {
    uint32_t num_metrics = _predict_config.getMetricNames().size();
    if (num_metrics != 1) {
      throw std::invalid_argument(
          "Validation-based early stopping only supports the use of one "
          "metric, passed in " +
          std::to_string(num_metrics) + " metrics.");
    }
  }

  EarlyStopValidation(dataset::BoltDatasetPtr validation_data,
                      dataset::BoltDatasetPtr validation_labels,
                      PredictConfig predict_config, uint32_t patience = 3)
      : EarlyStopValidation({std::move(validation_data)},
                            std::move(validation_labels),
                            std::move(predict_config), patience) {}

  void onTrainBegin(BoltGraph& model, TrainConfig& train_config) final {
    (void)model;
    (void)train_config;

    std::string metric_name = _predict_config.getMetricNames()[0];

    // setting these onTrainBegin allows callback instances to be reused
    _epochs_since_best = 0;
    _should_stop_training = false;
    _should_minimize =
        MetricUtils::getMetricByName(metric_name)->smallerIsBetter();
    _best_validation_metric = _should_minimize
                                  ? std::numeric_limits<double>::min()
                                  : std::numeric_limits<double>::max();
  }

  void onEpochEnd(BoltGraph& model, TrainConfig& train_config) final {
    (void)train_config;

    std::string metric_name = _predict_config.getMetricNames()[0];

    double metric_val =
        model.predict(_validation_data, _validation_labels, _predict_config)
            .first[metric_name];

    if (isImprovement(metric_val)) {
      _best_validation_metric = metric_val;
      _epochs_since_best = 0;
      model.checkpointInMemory();
    } else {
      if (_epochs_since_best == _patience) {
        _should_stop_training = true;
      }
    }
  }

  void onTrainEnd(BoltGraph& model, TrainConfig& train_config) final {
    (void)train_config;
    model.loadCheckpointFromMemory();
  }

  bool shouldStopTraining() final { return _should_stop_training; }

 private:
  bool isImprovement(double metric_val) {
    if (_should_minimize) {
      return metric_val < _best_validation_metric;
    }
    return metric_val > _best_validation_metric;
  }

  dataset::BoltDatasetList _validation_data;
  dataset::BoltDatasetPtr _validation_labels;
  uint32_t _patience;
  PredictConfig _predict_config;

  bool _should_stop_training;
  uint32_t _epochs_since_best;
  bool _should_minimize;
  double _best_validation_metric;
};

using EarlyStopValidationPtr = std::shared_ptr<EarlyStopValidation>;

}  // namespace thirdai::bolt