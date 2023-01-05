#pragma once

#include <memory>

namespace thirdai::bolt::train {

class TrainState {
 public:
  explicit TrainState(float learning_rate)
      : _learning_rate(learning_rate), _stop_training(false) {}

  static std::shared_ptr<TrainState> make(float learning_rate) {
    return std::make_shared<TrainState>(learning_rate);
  }

  float learningRate() const { return _learning_rate; }

  void updateLearningRate(float new_learning_rate) {
    _learning_rate = new_learning_rate;
  }

  bool isTrainingStopped() const { return _stop_training; }

  void stopTraining() { _stop_training = true; }

 private:
  float _learning_rate;
  bool _stop_training;
};

using TrainStatePtr = std::shared_ptr<TrainState>;

}  // namespace thirdai::bolt::train