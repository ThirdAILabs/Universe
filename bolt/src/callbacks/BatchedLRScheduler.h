#pragma once

#include "Callback.h"
#include <bolt/src/graph/ExecutionConfig.h>
#include <optional>
#include <string>

namespace thirdai::bolt {

class BatchedLRScheduler : public Callback {
 public:
  explicit BatchedLRScheduler(std::string save_loc,
                              const std::string& monitored_metric,
                              uint32_t n_bad_batches_before_update,
                              uint32_t n_total_lr_updates, double scaledown,
                              uint32_t warmup_batches)
      : _save_loc(std::move(save_loc)),
        _metric(makeMetric(monitored_metric)),
        _n_bad_batches_before_update(n_bad_batches_before_update),
        _n_total_lr_updates(n_total_lr_updates),
        _scaledown(scaledown),
        _warmup_batches(warmup_batches) {}

  void onBatchEnd(BoltGraph& model, TrainState& train_state) final {
    (void)model;
    if (_num_batches < _warmup_batches) {
      _num_batches++;
      return;
    }

    double cur_metric =
        train_state.getAllTrainBatchMetrics()[_metric->name()].back();

    if (cur_metric > _best_metric) {
      // Save model
      _best_metric = cur_metric;
    }

    if (cur_metric < _last_metric) {
      _n_bad_batches++;
      std::cout << "Cur metric: " << cur_metric
                << " is less than last metric of " << _last_metric
                << ". Incrementing n bad batches" << std::endl;
    } else {
      _n_bad_batches = 0;
    }

    if (_n_bad_batches > _n_bad_batches_before_update) {
      if (_n_lr_updates > _n_total_lr_updates) {
        train_state.stop_training = true;
      } else {
        std::cout << "Scaling down LR from " << train_state.learning_rate
                  << " to ";
        train_state.learning_rate /= _scaledown;
        std::cout << train_state.learning_rate
                  << ". Num Updates = " << _n_lr_updates << std::endl;
        _n_lr_updates++;
      }
    }

    _last_metric = cur_metric;
  }

 private:
  std::string _save_loc;
  std::shared_ptr<Metric> _metric;
  uint32_t _n_bad_batches_before_update;
  uint32_t _n_bad_batches = 0;
  uint32_t _n_total_lr_updates;
  uint32_t _n_lr_updates = 0;
  double _scaledown;
  double _best_metric = 0;
  double _last_metric = 0;
  uint32_t _warmup_batches;
  uint32_t _num_batches = 0;
};

}  // namespace thirdai::bolt