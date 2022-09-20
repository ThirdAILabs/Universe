#pragma once

#include "Callback.h"
#include <bolt/src/graph/Graph.h>
#include <cmath>
#include <functional>
#include <memory>
#include <optional>
#include <stdexcept>
#include <utility>

namespace thirdai::bolt {

class LRSchedule {
 public:
  virtual float getNextLR(float current_learning_rate, uint32_t epoch) = 0;
  virtual ~LRSchedule() = default;
};

using LRSchedulePtr = std::shared_ptr<LRSchedule>;

/**
 * @brief Decays the learning rate by a factor of gamma once the number of
 * epochs reaches one of the specified milestones.
 * @param gamma: multiplicative factor
 * @param milestones: epoch milestones
 *
 * Ex. If lr=0.01, gamma=0.5, milestones=[5,10], then
 * lr=0.01 for 1 <= epoch <= 4
 * lr=0.005 for 5 <= epoch <= 9
 * lr=0.0025 for epoch >= 10
 */
class MultiStepLR final : public LRSchedule {
 public:
  MultiStepLR(float gamma, std::vector<uint32_t> milestones)
      : _gamma(gamma), _milestones(std::move(milestones)) {}
  float getNextLR(float current_learning_rate, const uint32_t epoch) final {
    if (std::find(_milestones.begin(), _milestones.end(), epoch + 1) !=
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
 * @brief Schedules per-epoch learning rate using a multiplicative factor
 * @param gamma: multiplicative factor
 */
class MultiplicativeLR final : public LRSchedule {
 public:
  explicit MultiplicativeLR(float gamma) : _gamma(gamma) {}

  float getNextLR(float current_learning_rate, const uint32_t epoch) final {
    (void)epoch;
    return current_learning_rate * _gamma;
  }

 private:
  float _gamma;
};
using MultiplicativeLRPtr = std::shared_ptr<MultiplicativeLR>;

/**
 * @brief Schedules per-epoch learning rate using an exponential factor
 * @param gamma: exponentiation factor
 */
class ExponentialLR final : public LRSchedule {
 public:
  explicit ExponentialLR(float gamma) : _gamma(gamma) {}

  float getNextLR(float current_learning_rate, const uint32_t epoch) final {
    (void)epoch;
    return current_learning_rate * exp(-_gamma);
  }

 private:
  float _gamma;
};
using ExponentialLRPtr = std::shared_ptr<ExponentialLR>;

/**
 * @brief Schedules per-epoch learning rate using a lambda function.
 * @param schedule: A function pointer for scheduling the learning rate
 *        The function pointer has the following signature:
 *        float schedule(float learning_rate, uint32_t epoch);
 */
class LambdaSchedule : public LRSchedule {
 public:
  explicit LambdaSchedule(std::function<float(float, uint32_t)> schedule)
      : _schedule(std::move(schedule)) {}

  float getNextLR(float current_learning_rate, const uint32_t epoch) final {
    return _schedule(current_learning_rate, epoch);
  }

 private:
  std::function<float(float, uint32_t)> _schedule;
};
using LambdaSchedulePtr = std::shared_ptr<LambdaSchedule>;

/**
 * @brief This callback is intended to schedule learning rate changes during
 * training.
 * @param schedule: a LRSchedule pointer for scheduling the learning rate. The
 * schedule is eithe pre-set or is a custom lambda function.
 */
class LearningRateScheduler final : public Callback {
 public:
  explicit LearningRateScheduler(LRSchedulePtr schedule)
      : _schedule(std::move(schedule)), _final_learning_rate(std::nullopt) {}

  void onEpochEnd(BoltGraph& model, TrainState& train_state) final {
    (void)model;
    uint32_t current_epoch = train_state.epoch;

    float current_learning_rate = train_state.learning_rate;
    if (_schedule) {
      train_state.learning_rate =
          _schedule->getNextLR(current_learning_rate, current_epoch);
    }
  }

  void onTrainEnd(BoltGraph& model, TrainState& train_state) final {
    (void)model;
    _final_learning_rate = train_state.learning_rate;
  }

  float getFinalLR() const { return *_final_learning_rate; }

 private:
  LRSchedulePtr _schedule;

  // Tracking the final learning rate for testing purposes
  std::optional<float> _final_learning_rate;
};

using LearningRateSchedulerPtr = std::shared_ptr<LearningRateScheduler>;

}  // namespace thirdai::bolt