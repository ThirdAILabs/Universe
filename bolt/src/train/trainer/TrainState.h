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
  explicit TrainState(float learning_rate, uint32_t batches_in_dataset)
      : _learning_rate(learning_rate),
        _batches_in_dataset(batches_in_dataset),
        _stop_training(false) {}

  static std::shared_ptr<TrainState> make(float learning_rate,
                                          uint32_t batches_in_dataset) {
    return std::make_shared<TrainState>(learning_rate, batches_in_dataset);
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

  uint32_t batchesInDataset() const { return _batches_in_dataset; }

 private:
  float _learning_rate;
  uint32_t _batches_in_dataset;
  bool _stop_training;
};

using TrainStatePtr = std::shared_ptr<TrainState>;

}  // namespace thirdai::bolt