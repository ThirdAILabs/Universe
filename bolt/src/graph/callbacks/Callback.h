#pragma once

#include <memory>
#include <vector>

namespace thirdai::bolt {

class BoltGraph;
using BoltGraphPtr = std::shared_ptr<BoltGraph>;

class TrainState;

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
  Callback() {}

  // instead of passing in a model we could have a model be set in
  // the constructor. This would require some sort of lambda/factory
  // constructor. We can think about this later.
  virtual void onTrainBegin(BoltGraph& model, TrainState& train_state) {
    (void)model;
    (void)train_state;
  }

  virtual void onTrainEnd(BoltGraph& model, TrainState& train_state) {
    (void)model;
    (void)train_state;
  }

  virtual void onEpochBegin(BoltGraph& model, TrainState& train_state) {
    (void)model;
    (void)train_state;
  }

  virtual void onEpochEnd(BoltGraph& model, TrainState& train_state) {
    (void)model;
    (void)train_state;
  }

  // currently predict/train is not supported for onbatch.. functions. this
  // would require refactoring of the graph api thus is saved for a future
  // change. Currently throws a supported error message.
  virtual void onBatchBegin(BoltGraph& model, TrainState& train_state) {
    (void)model;
    (void)train_state;
  }

  virtual void onBatchEnd(BoltGraph& model, TrainState& train_state) {
    (void)model;
    (void)train_state;
  }

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

  void onTrainBegin(BoltGraph& model, TrainState& train_state) {
    for (const auto& callback : _callbacks) {
      callback->onTrainBegin(model, train_state);
    }
  }

  void onTrainEnd(BoltGraph& model, TrainState& train_state) {
    for (const auto& callback : _callbacks) {
      callback->onTrainEnd(model, train_state);
    }
  }

  void onEpochBegin(BoltGraph& model, TrainState& train_state) {
    for (const auto& callback : _callbacks) {
      callback->onEpochBegin(model, train_state);
    }
  }

  void onEpochEnd(BoltGraph& model, TrainState& train_state) {
    for (const auto& callback : _callbacks) {
      callback->onEpochEnd(model, train_state);
    }
  }

  void onBatchBegin(BoltGraph& model, TrainState& train_state) {
    for (const auto& callback : _callbacks) {
      callback->onBatchBegin(model, train_state);
    }
  }

  void onBatchEnd(BoltGraph& model, TrainState& train_state) {
    for (const auto& callback : _callbacks) {
      callback->onBatchEnd(model, train_state);
    }
  }

 private:
  std::vector<CallbackPtr> _callbacks;
};

}  // namespace thirdai::bolt