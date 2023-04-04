#pragma once

#include "Callback.h"
#include <bolt/src/graph/ExecutionConfig.h>
#include <optional>
#include <string>

namespace thirdai::bolt {

class ReduceLROnPlateau : public Callback {
 public:
  explicit ReduceLROnPlateau(const std::string& monitored_metric,
                             float factor = 0.2, uint32_t patience = 10,
                             uint32_t n_total_lr_updates = 10,
                             float min_delta = 0, uint32_t cooldown = 0,
                             bool verbose = false)
      : _metric(makeMetric(monitored_metric)),
        _factor(factor),
        _patience(patience),
        _n_total_lr_updates(n_total_lr_updates),
        _min_delta(min_delta),
        _cooldown(cooldown),
        _verbose(verbose),
        _last_metric(_metric->worst()) {
    assertGreaterThanZero(factor, "factor");
    assertGreaterThanZero(patience, "patience");
    assertGreaterThanZero(n_total_lr_updates, "n_total_lr_updates");
    assertGreaterThanOrEqualToZero(min_delta, "min_delta");
    assertGreaterThanOrEqualToZero(cooldown, "cooldown");
  }

  void onTrainBegin(BoltGraph& model, TrainState& train_state) final {
    (void)model;
    auto train_metrics = train_state.validation_metric_names;
    if (std::find(train_metrics.begin(), train_metrics.end(),
                  _metric->name()) == train_metrics.end()) {
      throw std::invalid_argument("ReduceLROnPlateau: Could not find metric " +
                                  _metric->name() +
                                  " in list of provided train metrics.");
    }
  }

  void onBatchEnd(BoltGraph& model, TrainState& train_state) final {
    (void)model;

    double cur_metric =
        train_state.getAllTrainBatchMetrics()[_metric->name()].back();

    if (isImprovement(cur_metric)) {
      _best_metric = cur_metric;
      _n_bad_batches = 0;
    } else {
      _n_bad_batches++;
    }

    if (_cooldown_count > 0) {
      _cooldown_count--;
      _n_bad_batches = 0;  // ignore any bad batches in cooldown
    }

    if (_n_bad_batches > _patience) {
      if (_n_lr_updates > _n_total_lr_updates) {
        train_state.stop_training = true;
      } else {
        std::cout << "Scaling down LR from " << train_state.learning_rate
                  << " to " << train_state.learning_rate * _factor
                  << ". Num Updates = " << _n_lr_updates << std::endl;
        train_state.learning_rate *= _factor;
        _n_lr_updates++;
        _cooldown_count = _cooldown;
        _n_bad_batches = 0;
      }
    }
  }

 private:
  template <typename T>
  void assertGreaterThanZero(T number, const std::string& var_name) {
    if (number <= 0) {
      throw std::invalid_argument(var_name + " should be > 0.");
    }
  }

  template <typename T>
  void assertGreaterThanOrEqualToZero(T number, const std::string& var_name) {
    if (number < 0) {
      throw std::invalid_argument(var_name + " should be >= 0.");
    }
  }

  bool isImprovement(double metric_value) {
    return std::abs(metric_value - _best_metric) >= _min_delta &&
           _metric->betterThan(metric_value, _best_metric);
  }

  std::shared_ptr<Metric> _metric;
  double _factor;
  uint32_t _patience;
  uint32_t _n_bad_batches = 0;
  uint32_t _n_total_lr_updates;
  uint32_t _n_lr_updates = 0;
  float _min_delta;
  uint32_t _cooldown;
  uint32_t _cooldown_count = 0;
  bool _verbose;
  double _best_metric;
};

}  // namespace thirdai::bolt