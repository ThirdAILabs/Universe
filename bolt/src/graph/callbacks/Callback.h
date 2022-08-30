#pragma once

#include <memory>
#include <vector>

namespace thirdai::bolt {

class BoltGraph;
using BoltGraphPtr = std::shared_ptr<BoltGraph>;

class TrainConfig;

class Callback;
using CallbackPtr = std::shared_ptr<Callback>;

/**
 * This class represents a generic Callback interface. Implementing this
 * interface allows you to call various methods at different steps during the
 * model training process. Functions may alter the model or train_config, state.
 *
 * Right now this callback is only used during training.
 */
class Callback {
 public:
  virtual void onTrainBegin(BoltGraph&, TrainConfig&){};

  virtual void onTrainEnd(BoltGraph&, TrainConfig&){};

  virtual void onEpochBegin(BoltGraph&, TrainConfig&){};

  virtual void onEpochEnd(BoltGraph&, TrainConfig&){};

  virtual void onBatchBegin(BoltGraph&, TrainConfig&){};

  virtual void onBatchEnd(BoltGraph&, TrainConfig&){};

  virtual bool shouldStopTraining() { return false; }

  virtual ~Callback() = default;
};

/**
 * This class serves as a helpful intermediary between models and callbacks by
 * dispatching calls to the stored callbacks.
 */
class CallbackList {
 public:
  explicit CallbackList(std::vector<CallbackPtr> callbacks)
      : _callbacks(std::move(callbacks)) {}

  void onTrainBegin(BoltGraph& model, TrainConfig& train_config) {
    for (const auto& callback : _callbacks) {
      callback->onTrainBegin(model, train_config);
    }
  }

  void onTrainEnd(BoltGraph& model, TrainConfig& train_config) {
    for (const auto& callback : _callbacks) {
      callback->onTrainEnd(model, train_config);
    }
  }

  void onEpochBegin(BoltGraph& model, TrainConfig& train_config) {
    for (const auto& callback : _callbacks) {
      callback->onEpochBegin(model, train_config);
    }
  }

  void onEpochEnd(BoltGraph& model, TrainConfig& train_config) {
    for (const auto& callback : _callbacks) {
      callback->onEpochEnd(model, train_config);
    }
  }

  void onBatchBegin(BoltGraph& model, TrainConfig& train_config) {
    for (const auto& callback : _callbacks) {
      callback->onBatchBegin(model, train_config);
    }
  }

  void onBatchEnd(BoltGraph& model, TrainConfig& train_config) {
    for (const auto& callback : _callbacks) {
      callback->onBatchEnd(model, train_config);
    }
  }

  bool shouldStopTraining() {
    return std::any_of(_callbacks.begin(), _callbacks.end(),
                       [&](const CallbackPtr& callback) {
                         return callback->shouldStopTraining();
                       });
  }

 private:
  std::vector<CallbackPtr> _callbacks;
};

}  // namespace thirdai::bolt