#pragma once

#include <memory>

namespace thirdai::bolt::train::trainer {

class State {
 public:
  explicit State(float learning_rate)
      : _learning_rate(learning_rate), _stop_training(false) {}

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

using StatePtr = std::shared_ptr<State>;

}  // namespace thirdai::bolt::train::trainer