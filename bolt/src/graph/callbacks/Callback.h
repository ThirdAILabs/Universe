#pragma once

#include <memory>
#include <vector>

namespace thirdai::bolt {

class BoltGraph;
using BoltGraphPtr = std::shared_ptr<BoltGraph>;

class Callback {
 public:
  void setModel(BoltGraphPtr model) { _model = std::move(model); }

  virtual void onTrainBegin(){};

  virtual void onTrainEnd(){};

  virtual void onEpochBegin(){};

  virtual void onEpochEnd(){};

  virtual void onBatchBegin(){};

  virtual void onBatchEnd(){};

  virtual ~Callback() = default;

 private:
  BoltGraphPtr _model;
};

using CallbackPtr = std::shared_ptr<Callback>;

class CallbackList {
 public:
  explicit CallbackList(std::vector<CallbackPtr> callbacks)
      : _callbacks(std::move(callbacks)) {}

  void onTrainBegin() {
    for (const auto& callback : _callbacks) {
      callback->onTrainBegin();
    }
  };

  void onTrainEnd() {
    for (const auto& callback : _callbacks) {
      callback->onTrainEnd();
    }
  };

  void onEpochBegin() {
    for (const auto& callback : _callbacks) {
      callback->onEpochBegin();
    }
  };

  void onEpochEnd() {
    for (const auto& callback : _callbacks) {
      callback->onEpochEnd();
    }
  };

  void onBatchBegin() {
    for (const auto& callback : _callbacks) {
      callback->onBatchBegin();
    }
  };

  void onBatchEnd() {
    for (const auto& callback : _callbacks) {
      callback->onBatchEnd();
    }
  };

 private:
  std::vector<CallbackPtr> _callbacks;
};

}  // namespace thirdai::bolt