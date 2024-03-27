#pragma once

#include <memory>

namespace thirdai::bolt {

/**
 * This class contains information about the current state of training that is
 * exposed to callbacks. We need this class so that callbacks have access to the
 * internal state of the call to train (otherwise they just have access to the
 * model object).
 */
class TrainState {
 public:
  explicit TrainState(float learning_rate)
      : _learning_rate(learning_rate),
        _stop_training(false),
        _steps_since_validation(0) {}

  static std::shared_ptr<TrainState> make(
      float learning_rate) {
    return std::make_shared<TrainState>(learning_rate);
  }

  /**
   * Returns the current learning rate.
   */
  float learningRate() const { return _learning_rate; }

  /**
   * Sets the learning rate to the provided learning rate.
   */
  void updateLearningRate(float new_learning_rate) {
    _learning_rate = new_learning_rate;
  }

  void incrementStepsSinceVal() { _steps_since_validation++; }

  void resetStepsSinceVal() { _steps_since_validation = 0; }
  
  bool compareStepsSinceVal(uint32_t validation_steps) const {
    return _steps_since_validation == validation_steps;
  }

  /**
   * Returns if the flag to stop training has been set.
   */
  bool isTrainingStopped() const { return _stop_training; }

  /**
   * Sets a flag which indicates to the trainer that training should be stoped.
   * This can be used to implement early stopping features of callbacks.
   */
  void stopTraining() { _stop_training = true; }

 private:
  float _learning_rate;
  bool _stop_training;
  uint32_t _steps_since_validation;
};

using TrainStatePtr = std::shared_ptr<TrainState>;

}  // namespace thirdai::bolt