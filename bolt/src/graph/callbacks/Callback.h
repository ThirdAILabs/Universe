#pragma once

#include <memory>
#include <vector>

namespace thirdai::bolt {

class BoltGraph;
using BoltGraphPtr = std::shared_ptr<BoltGraph>;

/**
 * This class represents a generic Callback interface. Implementing this
 * interface allows you to call various methods at different steps during the
 * model training process. Right now this callback is only used during training
 * but could easily be adapted for use during inference by adding the virtual
 * methods: onPredictBegin and onPredictEnd.
 *
 * This class has access to the associated BoltGraph object as it is set at the
 * beginning of train(..). Callbacks are allowed to modify it as needed.
 */
class Callback {
 public:
  void setModel(BoltGraph* model) { _model = model; }

  virtual void onTrainBegin(){};

  virtual void onTrainEnd(){};

  virtual void onEpochBegin(){};

  virtual void onEpochEnd(){};

  virtual void onBatchBegin(){};

  virtual void onBatchEnd(){};

  virtual bool shouldStopTraining() { return false; }

  virtual CallbackPtr makeCopy();

  virtual ~Callback() = default;

 protected:
  // TODO(nick, david): ideally this is a shared_ptr. We can pass in
  // shared_from_this() in the BoltGraph class to fix this but this requires a
  // shared_ptr to the object to already exist when using it, thus we'd have to
  // refactor all instances of BoltGraph to BoltGraphPtr which is another PR
  BoltGraph* _model = nullptr;
};

using CallbackPtr = std::shared_ptr<Callback>;

/**
 * This class serves as a helpful intermediary between models and callbacks by
 * dispatching calls to the stored callbacks.
 */
class CallbackList {
 public:
  explicit CallbackList(std::vector<CallbackPtr> callbacks)
      : _callbacks(std::move(callbacks)) {}

  void setModel(BoltGraph* model) {
    for (const auto& callback : _callbacks) {
      callback->setModel(model);
    }
  }

  void onTrainBegin() {
    for (const auto& callback : _callbacks) {
      callback->onTrainBegin();
    }
  }

  void onTrainEnd() {
    for (const auto& callback : _callbacks) {
      callback->onTrainEnd();
    }
  }

  void onEpochBegin() {
    for (const auto& callback : _callbacks) {
      callback->onEpochBegin();
    }
  }

  void onEpochEnd() {
    for (const auto& callback : _callbacks) {
      callback->onEpochEnd();
    }
  }

  void onBatchBegin() {
    for (const auto& callback : _callbacks) {
      callback->onBatchBegin();
    }
  }

  void onBatchEnd() {
    for (const auto& callback : _callbacks) {
      callback->onBatchEnd();
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