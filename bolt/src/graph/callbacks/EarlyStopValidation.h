#pragma once

#include "Callback.h"
#include <bolt/src/graph/ExecutionConfig.h>
#include <dataset/src/Datasets.h>

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
                      dataset::BoltTokenDatasetList validation_tokens,
                      dataset::BoltDatasetPtr validation_labels,
                      PredictConfig predict_config, uint32_t patience = 2)
      : _validation_data(std::move(validation_data)),
        _validation_tokens(std::move(validation_tokens)),
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
      : EarlyStopValidation({std::move(validation_data)}, {},
                            std::move(validation_labels),
                            std::move(predict_config), patience) {}

  void onTrainBegin() final {
    // setting these here allows callback instances to be reused
    _epochs_since_best = 0;
    _best_validation_metric = 0;  // TODO SET TO A APPROPRIATE INF VALUE
    _should_stop_training = false;
  }

  void onEpochEnd() final {
    std::string metric_name = _predict_config.getMetricNames()[0];

    double metric_val = _model
                            ->predict(_validation_data, _validation_tokens,
                                      _validation_labels, _predict_config)
                            .first[metric_name];

    // for a metric where smaller is better (like MeanSquaredError), negating
    // the metric value allows the maximization logic below to function like a
    // minimization objective
    if (MetricUtils::getMetricByName(metric_name)->smallerIsBetter()) {
      metric_val = -metric_val;
    }

    if (metric_val > _best_validation_metric) {
      _best_validation_metric = metric_val;
      _epochs_since_best = 0;
      _model->checkpointInMemory();
    } else {
      if (_patience == 0) {
        _should_stop_training = true;
      }
    }
  }

  void onTrainEnd() final { _model->loadCheckpointFromMemory(); }

  bool shouldStopTraining() final { return _should_stop_training; }

 private:
  bool isImprovement(double metric_val) {}

  dataset::BoltDatasetList _validation_data;
  dataset::BoltTokenDatasetList _validation_tokens;
  dataset::BoltDatasetPtr _validation_labels;
  uint32_t _patience;
  PredictConfig _predict_config;

  uint32_t _epochs_since_best;
  double _best_validation_metric;

  bool _should_stop_training;
};

using EarlyStopValidationPtr = std::shared_ptr<EarlyStopValidation>;

}  // namespace thirdai::bolt