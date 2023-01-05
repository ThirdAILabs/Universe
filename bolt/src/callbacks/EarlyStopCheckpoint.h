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
 * @brief This callback monitors a validation metric and gives users a means to
 * configure their model training based on that metric. It provides features
 * such as saving the best scoring model on the validation set, stopping
 * training early, adjusting the learning rate, and more.
 *
 * @param model_save_path The file path to save the model that scored the
 * best on the validation set.
 * @param monitored_metric Optional: The metric to monitor for early stopping.
 * If there is no metric specified we will use the validation metric
 * provided. We will throw an error if there are no tracked validation metrics,
 * if validation is not set up, or if there are multiple validation metrics.
 * @param patience The number of epochs with no improvement in previous
 * validation score after which we will evaluate whether to do one of two
 * things: 1) adjust the learning rate and continue training or 2) stop
 * training if we've changed the learning rate enough times.
 * @param max_lr_adjustments The maximum number of learning rate
 * adjustments allowed after a "patience" interval.
 * @param lr_multiplier Multiplier for the learning rate after a "patience"
 * interval.
 * @param min_delta The minimum change in the monitored quantity to qualify as
 * an improvement, i.e. an absolute change of less than min_delta, will count as
 * no improvement.
 * @param compare_against One of "best" or "prev". Determines whether to compare
 * against the best validation metric so far or the previous validation metric
 * recorded.
 * @param time_out Optional. Represents the total training time (in seconds)
 * after which the model will stop training. Rounds up to the nearest epoch.
 *
 * Based on the keras design found here:
 * https://keras.io/api/callbacks/early_stopping/
 */
class EarlyStopCheckpoint : public Callback {
 public:
  explicit EarlyStopCheckpoint(
      std::string model_save_path,
      std::optional<std::string> monitored_metric = std::nullopt,
      uint32_t patience = 2, uint32_t max_lr_adjustments = 2,
      float lr_multiplier = 0.5, double min_delta = 0,
      std::string compare_against = "prev",
      std::optional<double> time_out = std::nullopt)
      : _monitored_metric_name(std::move(monitored_metric)),
        _model_save_path(std::move(model_save_path)),
        _patience(patience),
        _max_lr_adjustments(max_lr_adjustments),
        _lr_multiplier(lr_multiplier),
        _min_delta(std::abs(min_delta)),
        _compare_against(std::move(compare_against)),
        _time_out(time_out),
        _n_consecutive_validation_drops(0),
        _n_lr_adjustments(0),
        _total_train_time(0) {
    if (_patience == 0) {
      throw std::invalid_argument("Patience should be greater than 0.");
    }

    if (_compare_against != "best" && _compare_against != "prev") {
      throw std::invalid_argument(
          "'compare_against' should be one of 'best' or 'prev'.");
    }

    if (lr_multiplier <= 0) {
      throw std::invalid_argument("'lr_multiplier' should be > 0.");
    }

    if (_time_out.has_value() and _time_out < 0) {
      throw std::invalid_argument("'time_out' cannot be negative.");
    }
  }

  void onTrainBegin(BoltGraph& model, TrainState& train_state) final {
    (void)model;

    // here we'll infer which metric to track for validation. we're doing this
    // on train begin because we'd like to infer which metric we're using in
    // case the user does not pass one in. we won't know until training begins

    // if the user passes in the metric we'll check for that one
    if (_monitored_metric_name.has_value()) {
      _metric = makeMetric(*_monitored_metric_name);
    } else {
      // if the user does not pass in a metric and there's only one available
      // we'll use that. otherwise we'll throw an error telling the user to
      // specify which one they'd like to use.
      auto validation_metrics = train_state.validation_metric_names;
      if (validation_metrics.size() != 1) {
        throw std::invalid_argument(
            "Cannot infer a validation metric to track for EarlyStopCheckpoint. This is either from "
            "not setting up validation, not passing in a validation metric, or "
            "passing in too many validation metrics.");
      }
      std::string inferred_metric_name = validation_metrics.front();
      _metric = makeMetric(inferred_metric_name);
    }
    _best_validation_score = _metric->worst();
    _previous_validation_score = _metric->worst();
  }

  void onEpochEnd(BoltGraph& model, TrainState& train_state) final {
    double metric_value =
        train_state.getValidationMetrics(_metric->name()).front();

    // if we've improved on the previous score
    if (isImprovement(metric_value)) {
      _n_consecutive_validation_drops = 0;
    } else {
      // if we have dropped the validation score from the previous score

      _n_consecutive_validation_drops += 1;

      // we know patience is not zero so this is safe
      if (_n_consecutive_validation_drops == _patience) {
        // stop training if we've lowered the learning rate enough times
        if (_n_lr_adjustments == _max_lr_adjustments) {
          train_state.stop_training = true;
          std::cout
              << "EarlyStopCheckpoint callback has made the maximum number of "
                 "learning rate adjustments. Terminating training.\n"
              << std::endl;
        } else {
          // otherwise we adjust the learning rate and reset the count on
          // consecutive validation drops
          std::cout
              << "EarlyStopCheckpoint: Validation metric has not improved for "
              << _n_consecutive_validation_drops
              << " consecutive epochs. Lowering the learning rate from "
              << train_state.learning_rate << " to "
              << train_state.learning_rate * _lr_multiplier << ".\n"
              << std::endl;
          train_state.learning_rate *= _lr_multiplier;
          _n_lr_adjustments += 1;
          _n_consecutive_validation_drops = 0;
        }
      }
    }

    // save the model if its the best so far
    if (_metric->betterThan(metric_value, _best_validation_score)) {
      _best_validation_score = metric_value;
      model.save(_model_save_path);
    }

    // stop training if we've timed out
    _total_train_time += train_state.epoch_times.back();
    if (_time_out.has_value() && _total_train_time > _time_out) {
      train_state.stop_training = true;
      std::cout << "EarlyStopCheckpoint callback terminating training due to "
                   "time_out.\n"
                << std::endl;
    }

    _previous_validation_score = metric_value;
  }

 private:
  bool isImprovement(double metric_value) {
    double score_to_compare_against = _previous_validation_score
                                          ? _compare_against == "prev"
                                          : _best_validation_score;
    return std::abs(metric_value - score_to_compare_against) >= _min_delta &&
           _metric->betterThan(metric_value, score_to_compare_against);
  }

  std::optional<std::string> _monitored_metric_name;
  std::shared_ptr<Metric> _metric;
  std::string _model_save_path;
  uint32_t _patience;
  uint32_t _max_lr_adjustments;
  double _lr_multiplier;
  double _min_delta;
  std::string _compare_against;
  std::optional<double> _time_out;

  uint32_t _n_consecutive_validation_drops;
  double _best_validation_score;
  double _previous_validation_score;
  uint32_t _n_lr_adjustments;
  double _total_train_time;
};

using EarlyStopCheckpointPtr = std::shared_ptr<EarlyStopCheckpoint>;

}  // namespace thirdai::bolt