#pragma once

#include <bolt/src/nn/model/Model.h>
#include <bolt/src/train/trainer/TrainState.h>
#include <memory>

namespace thirdai::bolt::train::callbacks {

class Callback {
 public:
  Callback() : model(nullptr) {}

  virtual void onTrainBegin() {}

  virtual void onTrainEnd() {}

  virtual void onEpochBegin() {}

  virtual void onEpochEnd() {}

  virtual void onBatchBegin() {}

  virtual void onBatchEnd() {}

  void setModel(nn::model::ModelPtr model);

  void setTrainState(TrainStatePtr train_state);

  virtual ~Callback() = default;

 protected:
  nn::model::ModelPtr model;
  TrainStatePtr train_state;
};

using CallbackPtr = std::shared_ptr<Callback>;

class CallbackList {
 public:
  CallbackList(std::vector<CallbackPtr> callbacks, nn::model::ModelPtr& model,
               TrainStatePtr& train_state);

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