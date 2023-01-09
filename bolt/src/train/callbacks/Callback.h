#pragma once

#include <bolt/src/nn/model/Model.h>
#include <bolt/src/train/metrics/Metric.h>
#include <bolt/src/train/trainer/TrainState.h>
#include <memory>

namespace thirdai::bolt::train::callbacks {

/**
 * Base class for a callback. When a callback is first passed into the train
 * method of a Trainer it is bound to that model and cannot be used on a
 * different model. The fields model, train_state, and history will be set with
 * the model it is bound to, the current train state for the trainer, and the
 * history of the trainer for that model respectively. The callback can use
 * these fields to access properties and functionality of the model, modify
 * current state of training such as the learning rate or early stopping, or
 * look at past values of train/validation metrics.
 */
class Callback {
 public:
  Callback() : model(nullptr) {}

  // Called at the begining of each call to train.
  virtual void onTrainBegin() {}

  // Called at the end of each call to train.
  virtual void onTrainEnd() {}

  // Called at the begining of each epoch.
  virtual void onEpochBegin() {}

  // Called at the end of each epoch.
  virtual void onEpochEnd() {}

  // Called when before each training batch.
  virtual void onBatchBegin() {}

  // Called after each training batch.
  virtual void onBatchEnd() {}

  /**
   * Binds the model to a callback.
   */
  void setModel(nn::model::ModelPtr model);

  /**
   * Sets the current train state the callback can access.
   */
  void setTrainState(TrainStatePtr train_state);

  /**
   * Binds the callback to the history (metric values).
   */
  void setHistory(metrics::HistoryPtr history);

  virtual ~Callback() = default;

 protected:
  nn::model::ModelPtr model;
  TrainStatePtr train_state;
  metrics::HistoryPtr history;
};

using CallbackPtr = std::shared_ptr<Callback>;

/**
 * Represents a list of callbacks. Binds callbacks to the provided model,
 * train_state, and history in its constructor. Calling any of the callback
 * methods will call the same method on all of its contained callbacks.
 */
class CallbackList {
 public:
  CallbackList(std::vector<CallbackPtr> callbacks, nn::model::ModelPtr& model,
               TrainStatePtr& train_state, metrics::HistoryPtr& history);

  void onTrainBegin();

  void onTrainEnd();

  void onEpochBegin();

  void onEpochEnd();

  void onBatchBegin();

  void onBatchEnd();

 private:
  std::vector<CallbackPtr> _callbacks;
};

}  // namespace thirdai::bolt::train::callbacks