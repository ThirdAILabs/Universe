#pragma once

#include "Callback.h"
#include <cstdint>
#include <memory>

namespace thirdai::bolt::train::callbacks {

/**
 * @brief This callback is intended to schedule learning rate changes during
 * training.
 * @param schedule: a LRSchedule pointer for scheduling the learning rate. The
 * schedule is either pre-set or is a custom lambda function.
 * @param batch_level_steps: If true then we'll adjust the learning rate using
 * batches as steps instead of epochs. Defaults to false.
 */
class LearningRateScheduler : public Callback {
 public:
  explicit LearningRateScheduler() : _epoch(0), _batch_cnt(0) {}

  virtual float getNextLR(float current_learning_rate, uint32_t step) = 0;

  void onEpochBegin() final {
    // reseting the batch count
    _batch_cnt = 0;

    if (!_batch_level_steps) {
      train_state->updateLearningRate(
          getNextLR(train_state->learningRate(), _epoch));
    }
    _epoch++;
  }

  void onBatchBegin() final {
    if (_batch_level_steps) {
      train_state->updateLearningRate(
          getNextLR(train_state->learningRate(), _batch_cnt));
    }
    _batch_cnt++;
  }

 private:
  uint32_t _epoch, _batch_cnt;

 protected:
  bool _batch_level_steps;
};

/**
 * @brief Schedules per-step learning rate linearly
 * @param start_factor: The multiplicative factor in the first epoch or batch
 * @param end_factor: TThe multiplicative factor at the end of the linear
 * changing process
 * @param total_iters: The number of iterations in which multiplicative factor
 * reaches to 1
 */
class LinearSchedule : public LearningRateScheduler {
 public:
  explicit LinearSchedule(float start_factor, float end_factor,
                          uint32_t total_iters, uint32_t batch_level_steps)
      : _start_factor(start_factor),
        _end_factor(end_factor),
        _total_iters(total_iters) {
    _batch_level_steps = batch_level_steps;
  }

  float getNextLR(float current_learning_rate, uint32_t step) final {
    if (step == 0) {
      // At the begining of step (or epoch) 0, we will have current learning
      // rate as the base learning rate

      // calculation of the lr change per step
      lr_change_per_step =
          current_learning_rate * (_end_factor - _start_factor) / _total_iters;

      return current_learning_rate * _start_factor;
    }
    if (step > _total_iters) {
      return current_learning_rate;
    }

    return current_learning_rate + lr_change_per_step;
  }

 private:
  float _start_factor, _end_factor, lr_change_per_step;
  uint32_t _total_iters;
};

}  // namespace thirdai::bolt::train::callbacks