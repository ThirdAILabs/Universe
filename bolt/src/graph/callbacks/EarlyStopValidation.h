#pragma once

#include "Callback.h"
#include <bolt/src/graph/ExecutionConfig.h>
#include <dataset/src/Datasets.h>

namespace thirdai::bolt {

/**
 * This class is intended to stop training early based on results from a given
 * validation set. After each epoch if the model performs worse on the
 * validation set than in the previous epoch, we decrement the "patience"
 * parameter by 1. When the patience value reaches 0 the model stops
 * training.
 *
 * This class always resets the model to use the best weights based on
 * validation performance. The purpose of this callback is to save potentially
 * harmful or unneeded training cycles, however one can always set patience
 * equal to the number of training epochs if the goal is to simply save and use
 * the best model from a training job.
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
        _predict_config(std::move(predict_config)),
        _best_validation_metric(0),
        _last_validation_metric(0),
        _should_stop_training(false) {
    uint32_t num_metrics = _predict_config.getNumMetricsTracked();
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

  void onEpochEnd() final {
    // we can access element 0 since we previously asserted having one metric
    std::string metric_name = _predict_config.getMetricNames()[0];

    if (!_model->getBatchSize().has_value()) {
      throw std::invalid_argument(
          "Misuse of EarlyStopValidation Callback. Model is not prepared for "
          "batch processing.");
    }

    uint32_t train_batch_size = _model->getBatchSize().value();

    _model->cleanupAfterBatchProcessing();

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

    if (metric_val < _last_validation_metric) {
      _patience--;
      if (_patience == 0) {
        _should_stop_training = true;
      }
    } else if (metric_val > _best_validation_metric) {
      _best_validation_metric = metric_val;
      _model->save(BEST_MODEL_SAVE_LOCATION);
    }
    _last_validation_metric = metric_val;

    _model->prepareToProcessBatches(train_batch_size,
                                    /* use_sparsity = */ true);
  }

  void onTrainEnd() final {
    *_model = *BoltGraph::load(BEST_MODEL_SAVE_LOCATION);
  }

  bool shouldStopTraining() final { return _should_stop_training; }

 private:
  dataset::BoltDatasetList _validation_data;
  dataset::BoltTokenDatasetList _validation_tokens;
  dataset::BoltDatasetPtr _validation_labels;
  uint32_t _patience;
  PredictConfig _predict_config;

  double _best_validation_metric;
  double _last_validation_metric;

  bool _should_stop_training;

  const std::string BEST_MODEL_SAVE_LOCATION = ".bestModel";
};

using EarlyStopValidationPtr = std::shared_ptr<EarlyStopValidation>;

}  // namespace thirdai::bolt