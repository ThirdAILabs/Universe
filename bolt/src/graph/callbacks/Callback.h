#pragma once

#include <memory>
#include <vector>

namespace thirdai::bolt {

class Callback {
 public:
  virtual void onTrainBegin(){};

  virtual void onTrainEnd(){};

  virtual void onEpochBegin(){};

  virtual void onEpochEnd(){};

  virtual void onBatchBegin(){};

  virtual void onBatchEnd(){};
};

using CallbackPtr = std::shared_ptr<Callback>;

class CallbackList {
 public:
  CallbackList(std::vector<CallbackPtr> callbacks) : _callbacks(callbacks) {}

  void onTrainBegin() {
    for (auto callback : _callbacks) {
      callback->onTrainBegin();
    }
  };

  void onTrainEnd() {
    for (auto callback : _callbacks) {
      callback->onTrainEnd();
    }
  };

  void onEpochBegin() {
    for (auto callback : _callbacks) {
      callback->onEpochBegin();
    }
  };

  void onEpochEnd() {
    for (auto callback : _callbacks) {
      callback->onEpochEnd();
    }
  };

  void onBatchBegin() {
    for (auto callback : _callbacks) {
      callback->onBatchBegin();
    }
  };

  void onBatchEnd() {
    for (auto callback : _callbacks) {
      callback->onBatchEnd();
    }
  };

 private:
  std::vector<CallbackPtr> _callbacks;
};

}  // namespace thirdai::bolt