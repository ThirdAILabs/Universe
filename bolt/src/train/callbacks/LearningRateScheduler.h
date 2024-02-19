#pragma once

#include "Callback.h"
#include <algorithm>
#include <cassert>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <variant>
#include <vector>

// There are issues including <cmath> to get M_PI on visual studio.
// https://stackoverflow.com/questions/6563810/m-pi-works-with-math-h-but-not-with-cmath-in-visual-studio
#define _USE_MATH_DEFINES
#include <math.h>  // NOLINT (clang-tidy wants <cmath>)

using state_type = std::unordered_map<
    std::string,
    std::variant<uint32_t, float, bool, std::string, std::vector<uint32_t>>>;

namespace thirdai::bolt::callbacks {
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
  explicit LearningRateScheduler(state_type& state) : _state(std::move(state)) {
    try {
      _epoch = std::get<uint32_t>(_state.at("_epoch"));
      _batch_cnt = std::get<uint32_t>(_state.at("_batch_cnt"));
      _batch_level_steps = std::get<bool>(_state.at("_batch_level_steps"));
      if (_state.find("_learning_rate") != _state.end()) {
        train_state->updateLearningRate(
            std::get<float>(state.at("_learning_rate")));
      }
    } catch (const std::exception& e) {
      std::cerr << e.what() << '\n';
    }
  }

  virtual float getNextLR(float current_learning_rate, uint32_t step) = 0;

  virtual void _update_state() = 0;

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

  virtual state_type get_state() {
    _state.clear();
    _state.emplace(std::make_pair("_epoch", _epoch));
    _state.emplace(std::make_pair("_batch_cnt", _batch_cnt));
    _state.emplace(std::make_pair("_batch_level_steps", _batch_level_steps));
    _state.emplace(
        std::make_pair("_learning_rate", train_state->learningRate()));
    _update_state();
    return _state;
  }

 protected:
  uint32_t _epoch, _batch_cnt;
  bool _batch_level_steps;
  state_type _state;
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
  explicit LinearSchedule(state_type& state) : LearningRateScheduler(state) {
    try {
      _start_factor = std::get<float>(_state.at("_start_factor"));
      _end_factor = std::get<float>(_state.at("_end_factor"));
      _lr_change_per_step = std::get<float>(_state.at("_lr_change_per_step"));
      _total_iters = std::get<uint32_t>(_state.at("_total_iters"));
      _update_state();
    } catch (const std::exception& e) {
      std::cerr << e.what() << '\n';
    }
  }

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

  void _update_state() final {
    _state.emplace(std::make_pair("_start_factor", _start_factor));
    _state.emplace(std::make_pair("_end_factor", _end_factor));
    _state.emplace(std::make_pair("_lr_change_per_step", _lr_change_per_step));
    _state.emplace(std::make_pair("_total_iters", _total_iters));
  }

 private:
  float _start_factor, _end_factor, _lr_change_per_step;
  uint32_t _total_iters;
};

/**
 * @brief Decays the learning rate by a factor of gamma once the number of
 * steps reaches one of the specified milestones.
 * @param gamma: multiplicative factor
 * @param milestones: step milestones
 * @param batch_level_steps: If true then we'll adjust the learning rate using
 * batches as steps instead of epochs. Defaults to false.
 */

class MultiStepLR final : public LearningRateScheduler {
 public:
  MultiStepLR(float gamma, std::vector<uint32_t>& milestones,
              bool batch_level_steps = false)
      : LearningRateScheduler(batch_level_steps),
        _gamma(gamma),
        _milestones(std::move(milestones)) {}
  explicit MultiStepLR(state_type& state) : LearningRateScheduler(state) {
    try {
      _gamma = std::get<float>(_state.at("_gamma"));
      _milestones = std::get<std::vector<uint32_t>>(_state.at("_milestones"));
      _update_state();
    } catch (const std::exception& e) {
      std::cerr << e.what() << '\n';
    }
  }
  float getNextLR(float current_learning_rate, uint32_t step) final {
    if (std::find(_milestones.begin(), _milestones.end(), step) !=
        _milestones.end()) {
      return current_learning_rate * _gamma;
    }
    return current_learning_rate;
  }

  void _update_state() final {
    _state.emplace(std::make_pair("_gamma", _gamma));
    _state.emplace(std::make_pair("_milestones", _milestones));
  }

 private:
  float _gamma;
  std::vector<uint32_t> _milestones;
};

/**
 * @brief Schedules per-step learning rate using a cosine annealing schedule
 * @param initial_restart_iter: Number of iterations before the first restart.
 * @param iter_restart_multiplicative_factor: Factor by which
 * _current_restart_iter gets multiplied after a restart. Default: 1
 * @param min_lr: Minimum learnign rate. Default: 0.0
 * @param batch_level_steps: If true then we'll adjust the learning rate using
 * batches as steps instead of epochs. Defaults to false.
 */

class CosineAnnealingWarmRestart final : public LearningRateScheduler {
 public:
  CosineAnnealingWarmRestart(float min_lr, float max_lr,
                             uint32_t steps_until_restart,
                             uint32_t linear_warmup_steps = 0,
                             uint32_t steps_until_restart_scaling_factor = 1,
                             bool batch_level_steps = true)
      : LearningRateScheduler(batch_level_steps),
        _min_lr(min_lr),
        _max_lr(max_lr),
        _steps(0),
        _steps_until_restart(steps_until_restart),
        _steps_until_restart_scaling_factor(steps_until_restart_scaling_factor),
        _linear_warmup_steps(linear_warmup_steps) {
    if (max_lr <= min_lr) {
      throw std::invalid_argument(
          "max lr should be greater than min lr in Cosine LR schedule.");
    }

    if (_steps_until_restart == 0 || _steps_until_restart_scaling_factor == 0) {
      throw std::invalid_argument(
          "steps_until_restart and steps_until_restart_scaling_factor must "
          "be "
          "nonzero.");
    }
  }
  explicit CosineAnnealingWarmRestart(state_type& state)
      : LearningRateScheduler(state) {
    _min_lr = std::get<float>(_state.at("_min_lr"));
    _max_lr = std::get<float>(_state.at("_max_lr"));
    _steps = std::get<uint32_t>(_state.at("_steps"));
    _steps_until_restart =
        std::get<uint32_t>(_state.at("_steps_until_restart"));
    _steps_until_restart_scaling_factor =
        std::get<uint32_t>(_state.at("_steps_until_restart_scaling_factor"));
    _linear_warmup_steps =
        std::get<uint32_t>(_state.at("_linear_warmup_steps"));
    _update_state();
  }

  float getNextLR(float current_learning_rate, uint32_t step) final {
    (void)current_learning_rate, (void)step;

    if (_linear_warmup_steps > 0) {
      float next_lr = _min_lr + static_cast<float>(_steps) /
                                    _linear_warmup_steps * (_max_lr - _min_lr);
      _steps++;
      if (_steps == _linear_warmup_steps) {
        _linear_warmup_steps = 0;
        _steps = 0;
      }
      return next_lr;
    }

    float cosine_factor =
        1 + std::cos(static_cast<float>(_steps) / _steps_until_restart * M_PI);
    float next_lr = _min_lr + (_max_lr - _min_lr) * cosine_factor / 2;

    _steps++;
    if (_steps == _steps_until_restart) {
      _steps = 0;
      _steps_until_restart *= _steps_until_restart_scaling_factor;
    }

    return next_lr;
  }

  void _update_state() final {
    _state.emplace(std::make_pair("_min_lr", _min_lr));
    _state.emplace(std::make_pair("_max_lr", _max_lr));
    _state.emplace(std::make_pair("_steps", _steps));
    _state.emplace(
        std::make_pair("_steps_until_restart", _steps_until_restart));
    _state.emplace(std::make_pair("_steps_until_restart_scaling_factor",
                                  _steps_until_restart_scaling_factor));
    _state.emplace(
        std::make_pair("_linear_warmup_steps", _linear_warmup_steps));
  }

 private:
  float _min_lr, _max_lr;
  uint32_t _steps, _steps_until_restart, _steps_until_restart_scaling_factor,
      _linear_warmup_steps;
};
}  // namespace thirdai::bolt::callbacks