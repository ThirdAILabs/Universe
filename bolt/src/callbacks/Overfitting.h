#pragma once

#include "Callback.h"
#include <bolt/src/graph/ExecutionConfig.h>
#include <optional>
#include <string>

namespace thirdai::bolt {

/**
 * This callback overfits the specified training metric to a threshold and
 * stops once that threshold is reached. This is useful for things like
 * cold-start where we attain the best results when overfitting the unsupervised
 * data.
 *
 * TODO(david): Add an optional model save at the end of each epoch once the UDT
 * saving callback stuff is fixed.
 */
class Overfitting : public Callback {
 public:
  explicit Overfitting(const std::string& monitored_metric,
                       float train_metric_threshold = 0.97)
      : _metric(makeMetric(monitored_metric)),
        _threshold(train_metric_threshold) {
    if (!_metric->betterThan(_threshold, _metric->worst())) {
      throw std::invalid_argument("Invalid threshold " +
                                  std::to_string(_threshold) + " for metric " +
                                  _metric->name() + ".");
    }
  }

  void onTrainBegin(BoltGraph& model, TrainState& train_state) final {
    (void)model;
    auto train_metrics = train_state.getTrainMetricAggregator().getMetrics();
    for (auto& metric : train_metrics) {
      if (_metric->name() == metric->name()) {
        return;
      }
    }
    throw std::invalid_argument("Metric: " + _metric->name() +
                                " not found in training metrics.");
  }

  void onEpochEnd(BoltGraph& model, TrainState& train_state) final {
    (void)model;
    double metric_value =
        train_state.getTrainMetricValues(_metric->name()).back();

    if (metric_value > _threshold) {
      train_state.stop_training = true;
      std::cout << "Reached threshold " << _threshold << " for training metric "
                << _metric->name() << ", stopping training. " << std::endl;
    }
  }

 private:
  std::shared_ptr<Metric> _metric;
  float _threshold;
};

}  // namespace thirdai::bolt