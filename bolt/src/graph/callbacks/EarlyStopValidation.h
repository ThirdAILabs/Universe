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
 * from a given validation set.
 *
 * To configure the stopping threshold we pass the "patience" argument which
 * tells us how many extra epochs we'll train for without beating our previous
 * best validation metric.
 *
 * This model will always save the best model checkpoint to a file on disk. If
 * restore_best_weights is false it will also save the last model checkpoint.
 * Otherwise it will reset the model to its best checkpoint. The default
 * behavior is not to restore the best weights.
 */
class EarlyStopValidation : public Callback {
 public:
  EarlyStopValidation(std::vector<dataset::BoltDatasetPtr> validation_data,
                      dataset::BoltDatasetPtr validation_labels,
                      PredictConfig predict_config, uint32_t patience = 2,
                      bool restore_best_weights = false)
      : _validation_data(std::move(validation_data)),
        _validation_labels(std::move(validation_labels)),
        _predict_config(std::move(predict_config)),
        _patience(patience),
        _restore_best_weights(restore_best_weights) {
    uint32_t num_metrics = _predict_config.getMetricNames().size();
    if (num_metrics != 1) {
      throw std::invalid_argument(
          "Validation-based early stopping only supports the use of one "
          "metric, passed in " +
          std::to_string(num_metrics) + " metrics.");
    }
  }

  void onTrainBegin(BoltGraph& model, TrainConfig& train_config) final {
    (void)model;
    (void)train_config;

    std::string metric_name = _predict_config.getMetricNames()[0];

    // setting these onTrainBegin allows callback instances to be reused
    _epochs_since_best = 0;
    _should_stop_training = false;
    _should_minimize = makeMetric(metric_name)->smallerIsBetter();
    _best_validation_score = _should_minimize
                                 ? std::numeric_limits<double>::min()
                                 : std::numeric_limits<double>::max();
  }

  void onEpochEnd(BoltGraph& model, TrainConfig& train_config) final {
    (void)train_config;

    std::string metric_name = _predict_config.getMetricNames()[0];
    double metric_val =
        model.predict(_validation_data, _validation_labels, _predict_config)
            .first[metric_name];

    _epochs_since_best++;
    if (isImprovement(metric_val)) {
      _best_validation_score = metric_val;
      _epochs_since_best = 0;
      model.save(BEST_MODEL_SAVE_LOCATION);
    } else if (_epochs_since_best == _patience) {
      _should_stop_training = true;
    }
  }

  void onTrainEnd(BoltGraph& model, TrainConfig& train_config) final {
    (void)train_config;
    if (_restore_best_weights) {
      model = *BoltGraph::load(BEST_MODEL_SAVE_LOCATION);
    } else {
      model.save(LAST_MODEL_SAVE_LOCATION);
    }
  }

  bool shouldStopTraining() final { return _should_stop_training; }

 private:
  bool isImprovement(double metric_val) const {
    if (_should_minimize) {
      return metric_val < _best_validation_score;
    }
    return metric_val > _best_validation_score;
  }

  inline static std::string BEST_MODEL_SAVE_LOCATION = "checkpoint_best.model";
  inline static std::string LAST_MODEL_SAVE_LOCATION = "checkpoint_last.model";

  std::vector<dataset::BoltDatasetPtr> _validation_data;
  dataset::BoltDatasetPtr _validation_labels;
  PredictConfig _predict_config;
  uint32_t _patience;
  bool _restore_best_weights;

  bool _should_stop_training;
  uint32_t _epochs_since_best;
  bool _should_minimize;
  double _best_validation_score;
};

using EarlyStopValidationPtr = std::shared_ptr<EarlyStopValidation>;

}  // namespace thirdai::bolt