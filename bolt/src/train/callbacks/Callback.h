#pragma once

#include <bolt/src/nn/model/Model.h>
#include <bolt/src/train/metrics/Metric.h>
#include <bolt/src/train/trainer/TrainState.h>
#include <memory>

namespace thirdai::bolt::callbacks {

/**
 * Base class for a callback. When a callback is first passed into the train
 * method of a Trainer it is bound to that model and cannot be used on a
 * different model. The callback has access to the Model, the TrainState and the
 * metric History which gives all the computed values of all metrics. The
 * callback can use these fields to access properties and functionality of the
 * model, modify current state of training such as the learning rate or early
 * stopping, or look at past values of train/validation metrics.
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

  virtual void beforeUpdate() {}

  ModelPtr& getModel() { return model; }

  TrainStatePtr& getTrainState() { return train_state; }

  metrics::History getHistory() const { return *history; }

  /**
   * Sets the model field in the callback so it can access the model. Cannot be
   * called more than once on a given callback object, i.e. a callback cannot be
   * used for multiple models.
   */
  void setModel(ModelPtr model);

  /**
   * Sets the current train state the callback can access.
   */
  void setTrainState(TrainStatePtr train_state);

  /**
   * Sets the history field in the callback so that the callback can access
   * metrics computed during training and validation. Cannot be called more than
   * once on a given callback object, i.e. a callback cannot be used for
   * multiple histories for different trainers.
   */
  void setHistory(metrics::HistoryPtr history);

  virtual ~Callback() = default;

 protected:
  /**
   * The fields model, train_state, and history will be set with the model it is
   * bound to, the current train state for the trainer, and the history of the
   * trainer for that model respectively.
   */
  ModelPtr model;
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
  CallbackList(std::vector<CallbackPtr> callbacks, ModelPtr& model,
               TrainStatePtr& train_state, metrics::HistoryPtr& history);

  void onTrainBegin();

  void onTrainEnd();

  void onEpochBegin();

  void onEpochEnd();

  void onBatchBegin();

  void onBatchEnd();

  void beforeUpdate();

 private:
  std::vector<CallbackPtr> _callbacks;
};

}  // namespace thirdai::bolt::callbacks