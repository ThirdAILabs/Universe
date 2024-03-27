#include "gtest/gtest.h"
#include <bolt/src/train/callbacks/ReduceLROnPlateau.h>
#include <bolt/src/train/metrics/Metric.h>
#include <bolt/src/train/trainer/TrainState.h>

namespace thirdai::bolt::tests {

void runLRScheduleTest(float threshold, bool relative_threshold) {
  auto history = std::make_shared<metrics::History>();
  auto train_state = TrainState::make(/* learning_rate= */ 10);

  callbacks::ReduceLROnPlateau callback(
      /* metric= */ "acc", /* patience= */ 2, /* cooldown= */ 1,
      /* decay_factor= */ 0.5, /* threshold= */ threshold,
      /* relative_threshold= */ relative_threshold, /* maximize= */ true,
      /* min_lr= */ 3);

  callback.setHistory(history);
  callback.setTrainState(train_state);

  std::vector<std::pair<float, float>> metrics_and_expected_lrs = {
      {4, 10}, {3, 10}, {7, 10}, {6, 10}, {8, 5}, {8.25, 5}, {8.5, 5}, {9, 3}};

  for (auto [metric, expected_lr] : metrics_and_expected_lrs) {
    (*history)["acc"].push_back(metric);
    callback.onEpochEnd();
    ASSERT_EQ(train_state->learningRate(), expected_lr);
  }
}

TEST(ReduceLROnPlateauTests, TestRelativeThreshold) {
  runLRScheduleTest(/* threshold= */ 0.5, /* relative_threshold= */ true);
}

TEST(ReduceLROnPlateauTests, TestAbsoluteThreshold) {
  runLRScheduleTest(/* threshold= */ 2, /* relative_threshold= */ false);
}

}  // namespace thirdai::bolt::tests