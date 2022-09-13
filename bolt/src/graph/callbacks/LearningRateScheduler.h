
#include <bolt/src/graph/Graph.h>
#include <bolt/src/graph/callbacks/Callback.h>
#include <cmath>
#include <functional>
#include <memory>
#include <optional>
#include <stdexcept>
#include <utility>

namespace thirdai::bolt {

// Pre-set learning rate scheduling primitives that are
// commonly used in practice.
enum class SchedulingPrimitive {
  MultiStepLR,
  MultiplicativeLR,
  ExplonentialLR,
};

/**
 * The LRSchedulingConfig configures the scheduling function and provides
 * several preset scheduling primitives.
 *
 * The parameters scheduling_primitive must be provided at construction time.
 * Other parameters necessary for different primitives will be set using the
 * builder pattern.
 *
 */

class LRSchedulingConfig {
  using SchedulingFunctionPtr = float (LRSchedulingConfig::*)(float, uint32_t);

 public:
  static LRSchedulingConfig makeConfig(
      const std::string& scheduling_primitive) {
    SchedulingPrimitive primitive =
        getSchedulingPrimitive(scheduling_primitive);
    return LRSchedulingConfig(primitive);
  }

  LRSchedulingConfig& withParameters(
      std::unordered_map<std::string, std::vector<float>> parameters) {
    _parameters = std::move(parameters);
    return *this;
  }

  float schedule(float learning_rate, const uint32_t epoch) {
    return (this->*_scheduling_function)(learning_rate, epoch);
  }

 private:
  explicit LRSchedulingConfig(SchedulingPrimitive primitive)
      : _primitive(primitive), _scheduling_function(configureSchedule()) {}

  // Decays the current learning rate by a certain factor specified by
  // _parameters["factor"] every epoch
  float multiplicativeLR(float learning_rate, const uint32_t epoch) {
    (void)epoch;
    if (!_parameters.count("factor")) {
      throw std::invalid_argument(
          "The multiplicative learning rate scheduler must have a "
          "multiplicative factor.\n");
    }
    if (_parameters["factor"].size() > 1) {
      throw std::invalid_argument(
          "The multiplicative learning rate scheduler should only have one "
          "multiplicative factor.\n");
    }
    return learning_rate * _parameters["factor"][0];
  }

  // Decays the current learning rate by the "gamma" factor specified in
  // _parameters["gamma"] when each epoch milestone is reached.
  // Epoch milestones are set as _parameters["milestones"]
  float multistepLR(float learning_rate, const uint32_t epoch) {
    (void)epoch;
    if (!_parameters.count("milestones")) {
      throw std::invalid_argument(
          "The multi-step learning rate scheduler must have an array of "
          "milestones.\n");
    }
    if (!_parameters.count("gamma")) {
      throw std::invalid_argument(
          "The multi-step learning rate scheduler must have the gamma "
          "multiplicative factor.\n");
    }
    float gamma = _parameters["gamma"][0];
    std::vector<float> milestones = _parameters["milestones"];

    if (std::find(milestones.begin(), milestones.end(), epoch + 1) !=
        milestones.end()) {
      return learning_rate * gamma;
    }

    return learning_rate;
  }
  // Decays the learning rate by a factor of exp(-gamma) every epoch
  float exponentialLR(float learning_rate, const uint32_t epoch) {
    (void)epoch;
    if (!_parameters.count("gamma")) {
      throw std::invalid_argument(
          "The exponential learning rate scheduler must have the gamma "
          "exponentiation factor.\n");
    }
    if (_parameters["gamma"].size() > 1) {
      throw std::invalid_argument(
          "The exponential learning rate scheduler should only have one "
          "exponentiation factor.\n");
    }

    float gamma = _parameters["gamma"][0];
    return learning_rate * exp(-gamma);
  }

  SchedulingFunctionPtr configureSchedule() {
    switch (_primitive) {
      case SchedulingPrimitive::ExplonentialLR: {
        return &LRSchedulingConfig::exponentialLR;
      }
      case SchedulingPrimitive::MultiStepLR: {
        return &LRSchedulingConfig::multistepLR;
      }
      case SchedulingPrimitive::MultiplicativeLR: {
        return &LRSchedulingConfig::multiplicativeLR;
      }
      default: {
        throw std::invalid_argument("Invalid scheduling primitive.\n");
      }
    }
  }

  static SchedulingPrimitive getSchedulingPrimitive(
      const std::string& scheduling_primitive) {
    if (scheduling_primitive == "multistep-lr") {
      return SchedulingPrimitive::MultiStepLR;
    }
    if (scheduling_primitive == "exponential-lr") {
      return SchedulingPrimitive::ExplonentialLR;
    }
    if (scheduling_primitive == "multiplicative-lr") {
      return SchedulingPrimitive::MultiplicativeLR;
    }

    throw std::invalid_argument("Invalid Learning rate scheduler.\n");
  }

  SchedulingPrimitive _primitive;
  // Function pointer that schedules the learning rate change
  // during training. The scheduler function signature is as follows:
  //    float schedule(float learning_rate, uint32_t epoch);
  SchedulingFunctionPtr _scheduling_function;

  std::unordered_map<std::string, std::vector<float>> _parameters;
};

/**
 * @brief This callback is intended to schedule learning rate changes during
 * training.
 *
 * @param scheduling_config: configuration for the scheduler which provides
 * access to common pre-set scheduling primitives.
 *
 * @param schedule: a custom function pointer for scheduling the learning rate.
 * schedule function signature: float schedule(float learning_rate, uint32_t
 * epoch);
 */
class LearningRateScheduler final : public Callback {
 public:
  // Defaults to constant learning rate across epochs
  LearningRateScheduler()
      : _scheduling_config(std::nullopt),
        _lambda_scheduler(std::nullopt),
        _final_learning_rate(std::nullopt) {}

  explicit LearningRateScheduler(const LRSchedulingConfig& scheduling_config)
      : _scheduling_config(scheduling_config),
        _final_learning_rate(std::nullopt) {}

  explicit LearningRateScheduler(std::function<float(float, uint32_t)> schedule)
      : _lambda_scheduler(std::move(schedule)),
        _final_learning_rate(std::nullopt) {}

  void onEpochBegin(BoltGraph& model, TrainState& train_state) final {
    uint32_t current_epoch = model.getEpochCount();

    if (!current_epoch) {
      return;
    }
    float current_learning_rate = train_state.learning_rate;
    if (_scheduling_config) {
      train_state.learning_rate =
          _scheduling_config->schedule(current_learning_rate, current_epoch);
    } else if (!_scheduling_config && _lambda_scheduler) {
      train_state.learning_rate =
          (*_lambda_scheduler)(current_learning_rate, current_epoch);
    }
  }

  void onTrainEnd(BoltGraph& model, TrainState& train_state) final {
    (void)model;
    _final_learning_rate = train_state.learning_rate;
  }

  float getFinalLR() const { return *_final_learning_rate; }

 private:
  std::optional<LRSchedulingConfig> _scheduling_config;
  std::optional<std::function<float(float, uint32_t)>> _lambda_scheduler;

  // Tracking the final learning rate for testing purposes
  std::optional<float> _final_learning_rate;
};

using LearningRateSchedulerPtr = std::shared_ptr<LearningRateScheduler>;

}  // namespace thirdai::bolt