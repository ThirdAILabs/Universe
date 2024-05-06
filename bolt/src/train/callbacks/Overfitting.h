#pragma once

#include <bolt/src/train/callbacks/Callback.h>
#include <optional>
#include <stdexcept>

namespace thirdai::bolt::callbacks {

class Overfitting : public Callback {
 public:
  explicit Overfitting(std::string metric, float threshold = 0.97,
                       bool freeze_hash_tables = true, bool maximize = true,
                       std::optional<uint32_t> min_epochs = std::nullopt)
      : _metric(std::move(metric)),
        _threshold(threshold),
        _maximize(maximize),
        _freeze_hash_tables(freeze_hash_tables),
        _min_epochs(min_epochs) {}

  void onTrainBegin() final {
    if (_freeze_hash_tables) {
      model->freezeHashTables(/* insert_labels_if_not_found = */ true);
    }
  }

  void onEpochEnd() final {
    if (!history->count(_metric)) {
      throw std::invalid_argument("Unable to find metric: '" + _metric + "'.");
    }

    float last_value = history->at(_metric).back();

    if (isImprovment(last_value) && minEpochsReached()) {
      train_state->stopTraining();
    }
  }

 private:
  bool isImprovment(float last_value) const {
    return (_maximize && last_value >= _threshold) ||
           (!_maximize && last_value <= _threshold);
  }

  bool minEpochsReached() const {
    return model->epochs() >= _min_epochs.value_or(0);
  }

  std::string _metric;
  float _threshold;
  bool _maximize;
  bool _freeze_hash_tables;
  std::optional<uint32_t> _min_epochs;
};

}  // namespace thirdai::bolt::callbacks