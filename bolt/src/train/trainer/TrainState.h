#pragma once

#include <memory>

namespace thirdai::bolt::train {

/**
 * This class contains information about the current state of training that is
 * exposed to callbacks.
 */
class TrainState {
 public:
  explicit TrainState(float learning_rate)
      : _learning_rate(learning_rate), _stop_training(false) {}

  static std::shared_ptr<TrainState> make(float learning_rate) {
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
};

using TrainStatePtr = std::shared_ptr<TrainState>;

}  // namespace thirdai::bolt::train