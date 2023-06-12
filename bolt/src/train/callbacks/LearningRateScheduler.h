#pragma once

#include <bolt/src/train/callbacks/Callback.h>
#include <cstdint>
#include <memory>

namespace thirdai::bolt::train {

class LRSchedule {
 public:
  virtual float getNextLR(float current_learning_rate, uint32_t step) = 0;

  virtual ~LRSchedule() = default;
};

using LRSchedulePtr = std::shared_ptr<LRSchedule>;

/**
 * @brief Schedules per-step learning rate linearly
 * @param start_factor: The multiplicative factor in the first epoch or batch
 * @param end_factor: TThe multiplicative factor at the end of the linear
 * changing process
 * @param total_iters: The number of iterations in which multiplicative factor
 * reaches to 1
 */
class LinearSchedule : public LRSchedule {
 public:
  explicit LinearSchedule(float start_factor, float end_factor,
                          uint32_t total_iters)
      : _start_factor(start_factor),
        _end_factor(end_factor),
        _total_iters(total_iters) {}

  float getNextLR(float current_learning_rate, uint32_t step) final {
    if (step == 0) {
      // At the begining of step (or epoch) 0, we will have current learning
      // rate as the base learning rate

      // calculation of the common difference(d)
      d = current_learning_rate * (_end_factor - _start_factor) / _total_iters;

      return current_learning_rate * _start_factor;
    }
    if (step > _total_iters) {
      return current_learning_rate;
    }

    return current_learning_rate + d;
  }

 private:
  float _start_factor, _end_factor, d;
  uint32_t _total_iters;
};

using LinearSchedulePtr = std::shared_ptr<LinearSchedule>;

/**
 * @brief This callback is intended to schedule learning rate changes during
 * training.
 * @param schedule: a LRSchedule pointer for scheduling the learning rate. The
 * schedule is either pre-set or is a custom lambda function.
 * @param batch_level_steps: If true then we'll adjust the learning rate using
 * batches as steps instead of epochs. Defaults to false.
 */
class LearningRateScheduler final : public Callback {
 public:
  explicit LearningRateScheduler(LRSchedulePtr schedule,
                                 bool batch_level_steps = false)
      : _schedule(std::move(schedule)), _batch_level_steps(batch_level_steps) {}

  void onBatchBegin() final {
    if (_batch_level_steps) {
      train_state.learning_rate = _schedule->getNextLR(
          train_state.learning_rate, train_state.batch_cnt);
    }
  }

  void onEpochBegin() final {
    if (!_batch_level_steps) {
      train_state.learning_rate =
          _schedule->getNextLR(train_state.learning_rate, train_state.epoch);
    }
  }

 private:
  LRSchedulePtr _schedule;
  bool _batch_level_steps;
};

using LearningRateSchedulerPtr = std::shared_ptr<LearningRateScheduler>;

}  // namespace thirdai::bolt::train