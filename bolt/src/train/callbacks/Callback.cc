#include "Callback.h"
#include <stdexcept>

namespace thirdai::bolt::callbacks {

void Callback::setModel(ModelPtr model) {
  if (this->model && this->model != model) {
    throw std::runtime_error("Cannot bind callback to new model.");
  }
  this->model = std::move(model);
}

void Callback::setTrainState(TrainStatePtr train_state) {
  this->train_state = std::move(train_state);
}

void Callback::setHistory(metrics::HistoryPtr history) {
  this->history = std::move(history);
}

CallbackList::CallbackList(std::vector<CallbackPtr> callbacks, ModelPtr& model,
                           TrainStatePtr& train_state,
                           metrics::HistoryPtr& history)
    : _callbacks(std::move(callbacks)) {
  for (auto& callback : _callbacks) {
    callback->setModel(model);
    callback->setTrainState(train_state);
    callback->setHistory(history);
  }
}

void CallbackList::onTrainBegin() {
  for (auto& callback : _callbacks) {
    callback->onTrainBegin();
  }
}

void CallbackList::onTrainEnd() {
  for (auto& callback : _callbacks) {
    callback->onTrainEnd();
  }
}

void CallbackList::onEpochBegin() {
  for (auto& callback : _callbacks) {
    callback->onEpochBegin();
  }
}

void CallbackList::onEpochEnd() {
  for (auto& callback : _callbacks) {
    callback->onEpochEnd();
    std::cout << "ONEPOCHEND IN CALLBACKLIST" << std::endl;
  }
}

void CallbackList::onBatchBegin() {
  for (auto& callback : _callbacks) {
    callback->onBatchBegin();
  }
}

void CallbackList::onBatchEnd() {
  for (auto& callback : _callbacks) {
    callback->onBatchEnd();
  }
}

void CallbackList::beforeUpdate() {
  for (auto& callback : _callbacks) {
    callback->beforeUpdate();
  }
}

}  // namespace thirdai::bolt::callbacks