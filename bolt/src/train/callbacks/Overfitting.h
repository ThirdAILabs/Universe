#pragma once

#include <bolt/src/train/callbacks/Callback.h>
#include <stdexcept>

namespace thirdai::bolt::train::callbacks {

class Overfitting : public Callback {
 public:
  explicit Overfitting(std::string metric, float threshold = 0.97,
                       bool maximize = true)
      : _metric(std::move(metric)),
        _threshold(threshold),
        _maximize(maximize) {}

  void onEpochEnd() final {
    if (!history->count(_metric)) {
      throw std::invalid_argument("Unable to find metric: '" + _metric + "'.");
    }

    float last_value = history->at(_metric).back();

    if ((_maximize && last_value >= _threshold) ||
        (!_maximize && last_value <= _threshold)) {
      train_state->stopTraining();
    }
  }

 private:
  std::string _metric;
  float _threshold;
  bool _maximize;
};

}  // namespace thirdai::bolt::train::callbacks