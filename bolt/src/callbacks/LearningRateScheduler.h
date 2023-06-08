#pragma once

#include "Callback.h"
#include <bolt/src/graph/Graph.h>
#include <cmath>
#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <stdexcept>
#include <utility>

namespace thirdai::bolt {

class LRSchedule {
 public:
  virtual float getNextLR(float current_learning_rate, uint32_t step) = 0;

  virtual ~LRSchedule() = default;
};

using LRSchedulePtr = std::shared_ptr<LRSchedule>;

/**
 * @brief Decays the learning rate by a factor of gamma once the number of steps
 * reaches one of the specified milestones.
 * @param gamma: multiplicative factor
 * @param milestones: step milestones
 *
 * Ex. If lr=0.01, gamma=0.5, milestones=[5,10], then
 * lr=0.01 for 1 <= step <= 4
 * lr=0.005 for 5 <= step <= 9
 * lr=0.0025 for step >= 10
 */
class MultiStepLR final : public LRSchedule {
 public:
  MultiStepLR(float gamma, std::vector<uint32_t> milestones)
      : _gamma(gamma), _milestones(std::move(milestones)) {}
  float getNextLR(float current_learning_rate, const uint32_t step) final {
    if (std::find(_milestones.begin(), _milestones.end(), step + 1) !=
        _milestones.end()) {
      return current_learning_rate * _gamma;
    }
    return current_learning_rate;
  }

 private:
  float _gamma;
  std::vector<uint32_t> _milestones;
};
using MultiStepLRPtr = std::shared_ptr<MultiStepLR>;

/**
 * @brief Schedules per-step learning rate using a multiplicative factor
 * @param gamma: multiplicative factor
 */
class MultiplicativeLR final : public LRSchedule {
 public:
  explicit MultiplicativeLR(float gamma) : _gamma(gamma) {}

  float getNextLR(float current_learning_rate, const uint32_t step) final {
    (void)step;
    return current_learning_rate * _gamma;
  }

 private:
  float _gamma;
};
using MultiplicativeLRPtr = std::shared_ptr<MultiplicativeLR>;

/**
 * @brief Schedules per-step learning rate using an exponential factor
 * @param gamma: exponentiation factor
 */
class ExponentialLR final : public LRSchedule {
 public:
  explicit ExponentialLR(float gamma) : _gamma(gamma) {}

  float getNextLR(float current_learning_rate, const uint32_t step) final {
    (void)step;
    return current_learning_rate * exp(-_gamma);
  }

 private:
  float _gamma;
};
using ExponentialLRPtr = std::shared_ptr<ExponentialLR>;

/**
 * @brief Schedules per-step learning rate using a lambda function.
 * @param schedule: A function pointer for scheduling the learning rate
 *        The function pointer has the following signature:
 *        float schedule(float learning_rate, uint32_t step);
 */
class LambdaSchedule : public LRSchedule {
 public:
  explicit LambdaSchedule(std::function<float(float, uint32_t)> lambda)
      : _lambda(std::move(lambda)) {}

  float getNextLR(float current_learning_rate, const uint32_t step) final {
    return _lambda(current_learning_rate, step);
  }

 private:
  std::function<float(float, uint32_t)> _lambda;
};
using LambdaSchedulePtr = std::shared_ptr<LambdaSchedule>;

/**
 * @brief Schedules per-step learning rate linearly
 * @param start_factor: The multiplicative factor in the first epoch
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
      return current_learning_rate * _start_factor;
    }

    if (step > _total_iters) {
      return current_learning_rate;
    }

    return current_learning_rate *
           (1. + (_end_factor - _start_factor) /
                     (_start_factor * _total_iters +
                      (step - 1) * (_end_factor - _start_factor)));
  }

 private:
  float _start_factor, _end_factor;
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

  void onBatchEnd(BoltGraph& model, TrainState& train_state) final {
    (void)model;
    if (_batch_level_steps) {
      train_state.learning_rate = _schedule->getNextLR(
          train_state.learning_rate, train_state.batch_cnt);
    }
  }

  void onEpochEnd(BoltGraph& model, TrainState& train_state) final {
    (void)model;
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

}  // namespace thirdai::bolt