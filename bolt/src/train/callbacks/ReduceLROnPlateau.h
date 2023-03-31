#pragma once

#include <bolt/src/train/callbacks/Callback.h>
#include <limits>
#include <stdexcept>

namespace thirdai::bolt::train::callbacks {

class ReduceLROnPlateau final : public Callback {
 public:
  ReduceLROnPlateau(std::string metric, uint32_t patience, uint32_t cooldown,
                    float decay_factor, float threshold,
                    bool relative_threshold, bool maximize = true,
                    float min_lr = 0)
      : _metric(std::move(metric)),
        _patience(patience),
        _cooldown(cooldown),
        _decay_factor(decay_factor),
        _threshold(threshold),
        _relative_threshold(relative_threshold),
        _maximize(maximize),
        _min_lr(min_lr),
        _patience_remaining(patience),
        _cooldown_remaining(cooldown) {
    if (_maximize) {
      _best_metric = 0;
    } else {
      _best_metric = std::numeric_limits<float>::max();
    }
  }

  void onEpochEnd() final {
    if (!history->count(_metric)) {
      throw std::invalid_argument("Unable to find metric: '" + _metric + "'.");
    }
    float metric_val = history->at(_metric).back();

    // If there is an improvement then make sure patience is reset and update
    // best metric. Otherwise decrement the patience remaining.
    if (isImprovement(metric_val)) {
      _best_metric = metric_val;
      _patience_remaining = _patience;
    } else {
      _patience_remaining--;
    }

    // If we are in cooldown then decrement the counter for another step
    // completed. Also reset patience so that lr updates cannot be considered
    // until after the cooldown is complete.
    if (_cooldown_remaining > 0) {
      _cooldown_remaining--;
      _patience_remaining = _patience;
    }

    // If patience is exhausted then reset the cooldown and patience and udpate
    // the learning rate.
    if (_patience_remaining == 0) {
      updateLearningRate();
      _cooldown_remaining = _cooldown;
      _patience_remaining = _patience;
    }

    // Update history so that the learning rate can be tracked like a metric on
    // mlflow.
    (*history)["learning_rate"].push_back(train_state->learningRate());
  }

 private:
  bool isImprovement(float metric_val) const {
    if (_maximize) {
      float improvement_threshold = _relative_threshold
                                        ? (1 + _threshold) * _best_metric
                                        : _best_metric + _threshold;
      return metric_val > improvement_threshold;
    }
    float improvement_threshold = _relative_threshold
                                      ? (1 - _threshold) * _best_metric
                                      : _best_metric - _threshold;
    return metric_val < improvement_threshold;
  }

  void updateLearningRate() {
    float current_lr = train_state->learningRate();
    float new_lr = std::max(current_lr * _decay_factor, _min_lr);

    train_state->updateLearningRate(new_lr);
  }

  std::string _metric;

  const uint32_t _patience;
  const uint32_t _cooldown;
  const float _decay_factor;
  const float _threshold;
  const bool _relative_threshold;
  const bool _maximize;
  const float _min_lr;

  uint32_t _patience_remaining;
  uint32_t _cooldown_remaining;
  float _best_metric;
};

}  // namespace thirdai::bolt::train::callbacks