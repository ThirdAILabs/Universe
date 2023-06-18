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
 * @param Initial_restart_iter: Number of iterations before the first restart.
 * @param iter_restart_multiplicative_factor: Factor by which
 * current_restart_iter_ gets multiplied after a restart. Default: 1
 * @param min_lr: Minimum learnign rate. Default: 0.0
 * @param batch_level_steps: If true then we'll adjust the learning rate using
 * batches as steps instead of epochs. Defaults to false.
 */

class CosineAnnealingWarmRestart final : public LearningRateScheduler {
 public:
  explicit CosineAnnealingWarmRestart(
      uint32_t Initial_restart_iter = 4,
      uint32_t iter_restart_multiplicative_factor = 1, float min_lr = 0.0,
      bool batch_per_step = false)
      : LearningRateScheduler(batch_per_step),
        current_restart_iter_(Initial_restart_iter),
        current_iter_(0),
        iter_restart_multiplicative_factor_(iter_restart_multiplicative_factor),
        min_lr_(min_lr) {
    assert(Initial_restart_iter > 0 &&
           iter_restart_multiplicative_factor >= 1 && min_lr >= 0);
  }

  float getNextLR(float current_learning_rate, uint32_t step) final {
    if (step == 0) {
      // base learning rate will be the maximum learning rate.
      base_learning_rate_ = current_learning_rate;
    }

    float cosine_factor = std::pow(
        std::cos((static_cast<float>(current_iter_) / current_restart_iter_) *
                 M_PI / 2),
        2);

    float nextLR = min_lr_ + (base_learning_rate_ - min_lr_) * cosine_factor;

    current_iter_++;
    if (current_iter_ == current_restart_iter_) {
      current_iter_ = 0;
      current_restart_iter_ *= iter_restart_multiplicative_factor_;
    }

    return nextLR;
  }

 private:
  uint32_t current_restart_iter_, current_iter_,
      iter_restart_multiplicative_factor_;
  float min_lr_, base_learning_rate_;
};
}  // namespace thirdai::bolt::train::callbacks