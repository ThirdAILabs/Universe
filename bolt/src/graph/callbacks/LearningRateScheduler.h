
#include <bolt/src/graph/Graph.h>
#include <bolt/src/graph/callbacks/Callback.h>
#include <functional>
#include <memory>
#include <optional>
#include <utility>

namespace thirdai::bolt {


class LearningRateScheduler final : public Callback {
 public:
  LearningRateScheduler() : _schedule(std::nullopt) {}
  explicit LearningRateScheduler(
      const std::function<float(float, uint32_t)>& schedule)
      : _schedule(schedule) {}

  void onEpochBegin(BoltGraph& model, TrainState& train_state) final {
    if (_schedule.has_value()) {
      uint32_t current_epoch = model.getEpochCount();
      float current_learning_rate = train_state.learning_rate;

      train_state.learning_rate =
          (*_schedule)(current_learning_rate, current_epoch);
    }
  }

  void onEpochEnd(BoltGraph& model, TrainState& train_state) final {
    (void)model;
    (void)train_state;
  }

 private:
  // Function pointer that schedules the learning rate change
  // during training. The scheduler function signature is as follows:
  //    float schedule(float lr, uint32_t epoch).
  std::optional<std::function<float(float, uint32_t)>> _schedule;
};

}  // namespace thirdai::bolt