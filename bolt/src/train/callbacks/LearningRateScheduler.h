#pragma once

#include "Callback.h"
#include <algorithm>
#include <assert.h>
#include <cmath>
#include <cstdint>
#include <memory>
#include <vector>

namespace thirdai::bolt::train::callbacks {

/**
 * @brief This callback is intended to schedule learning rate changes during
 * training.
 * @param batch_level_steps: If true then we'll adjust the learning rate using
 * batches as steps instead of epochs. Defaults to false.
 */
class LearningRateScheduler : public Callback {
 public:
  explicit LearningRateScheduler(bool batch_level_steps)
      : _epoch(0), _batch_cnt(0), _batch_level_steps(batch_level_steps) {}

  virtual float getNextLR(float current_learning_rate, uint32_t step) = 0;

  void onEpochBegin() final {
    // resetting the batch count
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
  bool _batch_level_steps;
};

/**
 * @brief Schedules per-step learning rate linearly
 * @param start_factor: The multiplicative factor in the first epoch or batch
 * @param end_factor: The multiplicative factor at the end of the linear
 * changing process
 * @param total_iters: The number of iterations in which multiplicative factor
 * reaches to end_factor
 * @param batch_level_steps: If true then we'll adjust the learning rate using
 * batches as steps instead of epochs. Defaults to false.
 */
class LinearSchedule final : public LearningRateScheduler {
 public:
  explicit LinearSchedule(float start_factor, float end_factor,
                          uint32_t total_iters, bool batch_level_steps = false)
      : LearningRateScheduler(batch_level_steps),
        _start_factor(start_factor),
        _end_factor(end_factor),
        _total_iters(total_iters) {}

  float getNextLR(float current_learning_rate, uint32_t step) final {
    if (step == 0) {
      // calculation of the lr change per step
      _lr_change_per_step =
          current_learning_rate * (_end_factor - _start_factor) / _total_iters;

      return current_learning_rate * _start_factor;
    }
    if (step > _total_iters) {
      return current_learning_rate;
    }

    return current_learning_rate + _lr_change_per_step;
  }

 private:
  float _start_factor, _end_factor, _lr_change_per_step;
  uint32_t _total_iters;
};

/**
 * @brief Decays the learning rate by a factor of gamma once the number of steps
 * reaches one of the specified milestones.
 * @param gamma: multiplicative factor
 * @param milestones: step milestones
 * @param batch_level_steps: If true then we'll adjust the learning rate using
 * batches as steps instead of epochs. Defaults to false.
 */

class MultiStepLR final : public LearningRateScheduler {
 public:
  MultiStepLR(float gamma, std::vector<uint32_t> milestones,
              bool batch_level_steps = false)
      : LearningRateScheduler(batch_level_steps),
        _gamma(gamma),
        _milestones(std::move(milestones)) {}

  float getNextLR(float current_learning_rate, uint32_t step) final {
    if (std::find(_milestones.begin(), _milestones.end(), step) !=
        _milestones.end()) {
      return current_learning_rate * _gamma;
    }
    return current_learning_rate;
  }

 private:
  float _gamma;
  std::vector<uint32_t> _milestones;
};

/**
 * @brief Schedules per-step learning rate using a cosine annealing schedule
 * @param T_0: Number of iterations before the first restart.
 * @param T_mul: Factor by which T_i increases after a restart. Default: 1
 * @param eta_min: Minimum learnign rate. Default: 0.0
 * @param batch_level_steps: If true then we'll adjust the learning rate using
 * batches as steps instead of epochs. Defaults to false.
 */

class CosineAnnealingWarmRestart final : public LearningRateScheduler {
 public:
  explicit CosineAnnealingWarmRestart(uint32_t T_0, uint32_t T_mul = 1,
                                      float eta_min = 0.0,
                                      bool batch_per_step = false)
      : LearningRateScheduler(batch_per_step),
        T_i_(T_0),
        T_mul_(T_mul),
        T_curr_(0),
        eta_min_(eta_min) {
    assert(T_0 > 0 && T_mul >= 1 && eta_min >= 0);
  }

  float getNextLR(float current_learning_rate, uint32_t step) final {
    if (step == 0) {
      // base learning rate will be the maximum learning rate.
      base_learning_rate_ = current_learning_rate;
    }

    float cosine_factor =
        1 + std::cos((static_cast<float>(T_curr_) / T_i_) * M_PI);

    float nextLR =
        eta_min_ + (base_learning_rate_ - eta_min_) * cosine_factor / 2;

    T_curr_++;
    if (T_curr_ >= T_i_) {
      T_curr_ -= T_i_;
      T_i_ *= T_mul_;
    }

    return nextLR;
  }

 private:
  uint32_t T_i_, T_mul_, T_curr_;
  float eta_min_, base_learning_rate_;
};
}  // namespace thirdai::bolt::train::callbacks