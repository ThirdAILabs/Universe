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
                              uint32_t n_total_lr_updates, double scaledown)
      : _save_loc(std::move(save_loc)),
        _metric(makeMetric(monitored_metric)),
        _n_bad_batches_before_update(n_bad_batches_before_update),
        _n_total_lr_updates(n_total_lr_updates),
        _scaledown(scaledown) {}

  void onBatchEnd(BoltGraph& model, TrainState& train_state) final {
    (void)model;
    double cur_metric =
        train_state.getAllTrainBatchMetrics()[_metric->name()].back();

    if (cur_metric > _best_metric) {
      train_state.stop_training = true;
      std::cout << "Reached threshold " << _threshold << " for training metric "
                << _metric->name() << ", stopping training. " << std::endl;
    }
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
};

}  // namespace thirdai::bolt