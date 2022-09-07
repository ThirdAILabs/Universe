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
 * model training process. Functions may alter the model state.
 *
 * Right now this callback is only used during training.
 *
 * TODO(david): lets make this a state machine where we assert that previous
 * steps are called before moving to the next stage. See Node.h getState() and
 * other methods for reference.
 */
class Callback {
 public:
  // TODO(david): instead of passing in a model we could have a model be set in
  // the constructor. This would require some sort of lambda/factory
  // constructor. We can think about this later.
  virtual void onTrainBegin(BoltGraph& model) { (void)model; };

  virtual void onTrainEnd(BoltGraph& model) { (void)model; };

  virtual void onEpochBegin(BoltGraph& model) { (void)model; };

  virtual void onEpochEnd(BoltGraph& model) { (void)model; };

  virtual void onBatchBegin(BoltGraph& model) { (void)model; };

  virtual void onBatchEnd(BoltGraph& model) { (void)model; };

  // TODO(david): semantically this is a little odd here, ideally we don't add
  // new functions every time we make a new callback. One alternative is to keep
  // a "training state" struct that we pass in to the callbacks which can be
  // changed/used during training.
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

  void onTrainBegin(BoltGraph& model) {
    for (const auto& callback : _callbacks) {
      callback->onTrainBegin(model);
    }
  }

  void onTrainEnd(BoltGraph& model) {
    for (const auto& callback : _callbacks) {
      callback->onTrainEnd(model);
    }
  }

  void onEpochBegin(BoltGraph& model) {
    for (const auto& callback : _callbacks) {
      callback->onEpochBegin(model);
    }
  }

  void onEpochEnd(BoltGraph& model) {
    for (const auto& callback : _callbacks) {
      callback->onEpochEnd(model);
    }
  }

  void onBatchBegin(BoltGraph& model) {
    for (const auto& callback : _callbacks) {
      callback->onBatchBegin(model);
    }
  }

  void onBatchEnd(BoltGraph& model) {
    for (const auto& callback : _callbacks) {
      callback->onBatchEnd(model);
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