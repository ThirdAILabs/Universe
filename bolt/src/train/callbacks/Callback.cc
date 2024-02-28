#include "Callback.h"
#include <bolt/src/train/callbacks/LearningRateScheduler.h>
#include <archive/src/Archive.h>
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

ar::ConstArchivePtr Callback::toArchve() const {
  throw std::runtime_error("This callback does not support serialization.");
}

std::shared_ptr<Callback> Callback::fromArchive(const ar::Archive& archive) {
  std::string type = archive.str("type");

  if (type == LinearSchedule::type()) {
    return std::make_shared<LinearSchedule>(archive);
  }
  if (type == MultiStepLR::type()) {
    return std::make_shared<MultiStepLR>(archive);
  }
  if (type == CosineAnnealingWarmRestart::type()) {
    return std::make_shared<CosineAnnealingWarmRestart>(archive);
  }

  throw std::invalid_argument("Unrecognized callback type '" + type +
                              "' in fromArchive.");
}

void Callback::save_stream(std::ostream& ostream) const {
  ar::serialize(toArchve(), ostream);
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