#pragma once

#include <memory>
#include <vector>

namespace thirdai::bolt {

class BoltGraph;
using BoltGraphPtr = std::shared_ptr<BoltGraph>;

class Callback {
 public:
  Callback() {}

  void setModel(BoltGraph* model) { _model = model; }

  virtual void onTrainBegin(){};

  virtual void onTrainEnd(){};

  virtual void onEpochBegin(){};

  virtual void onEpochEnd(){};

  virtual void onBatchBegin(){};

  virtual void onBatchEnd(){};

  virtual bool shouldStopTraining() { return false; }

  virtual ~Callback() = default;

 protected:
  BoltGraph* _model;
};

using CallbackPtr = std::shared_ptr<Callback>;

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