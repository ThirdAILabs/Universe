#include <bolt/src/train/callbacks/LearningRateScheduler.h>
#include <gtest/gtest.h>

namespace thirdai::bolt::tests {

static constexpr float MIN_LR = 1, MAX_LR = 2;

std::vector<float> getLRSchedule(uint32_t linear_warmup_steps) {
  callbacks::CosineAnnealingWarmRestart cos_schedule(
      /*min_lr=*/MIN_LR, /*max_lr=*/MAX_LR, /*steps_until_restart=*/3,
      /*linear_warmup_steps=*/linear_warmup_steps,
      /*steps_until_restart_scaling_factor=*/2);

  std::vector<float> schedule;
  // 2 cycles, 3 steps until first restart, 6 steps until second restart.
  for (uint32_t i = 0; i < linear_warmup_steps + 9; i++) {
    schedule.push_back(cos_schedule.getNextLR(0, 0));
  }

  return schedule;
}

void checkCosCyles(const std::vector<float>& schedule) {
  ASSERT_EQ(schedule.size(), 9);

  // The schedule should decay the learning rate for the first 3 steps (first
  // cycle). Then the schedule will restart and increase the cycle to 6 steps.
  // It will then decay for 6 steps (second cycle). Because of the  nature of
  // the cosine lr schedule, the lr will decay slower at the beginning and end
  // of a cycle than in the middle.

  // First cycle
  ASSERT_EQ(schedule[0], MAX_LR);
  float cycle1_step1 = schedule[0] - schedule[1];
  float cycle1_step2 = schedule[1] - schedule[2];
  float cycle1_step3 = schedule[2] - MIN_LR;
  ASSERT_EQ(cycle1_step1, cycle1_step3);
  ASSERT_LE(cycle1_step1, cycle1_step2);

  // Second cycle
  ASSERT_EQ(schedule[3], MAX_LR);
  float cycle2_step1 = schedule[3] - schedule[4];
  float cycle2_step2 = schedule[4] - schedule[5];
  float cycle2_step3 = schedule[5] - schedule[6];
  float cycle2_step4 = schedule[6] - schedule[7];
  float cycle2_step5 = schedule[7] - schedule[8];
  float cycle2_step6 = schedule[8] - MIN_LR;
  ASSERT_EQ(cycle2_step1, cycle2_step6);
  ASSERT_EQ(cycle2_step2, cycle2_step5);
  ASSERT_EQ(cycle2_step3, cycle2_step4);
  ASSERT_LE(cycle2_step1, cycle2_step2);
  ASSERT_LE(cycle2_step2, cycle2_step3);
}

TEST(CosineLRScheduleTests, WithLinearWarmup) {
  auto schedule = getLRSchedule(/*linear_warmup_steps=*/4);

  ASSERT_EQ(schedule.size(), 13);

  // Check linear warmup
  ASSERT_EQ(schedule[0], MIN_LR);
  ASSERT_EQ(schedule[1], MIN_LR + 0.25);
  ASSERT_EQ(schedule[2], MIN_LR + 0.5);
  ASSERT_EQ(schedule[3], MIN_LR + 0.75);
  ASSERT_EQ(schedule[4], MAX_LR);

  checkCosCyles({schedule.begin() + 4, schedule.end()});
}

TEST(CosineLRScheduleTests, WithoutLinearWarmup) {
  auto schedule = getLRSchedule(/*linear_warmup_steps=*/0);

  ASSERT_EQ(schedule.size(), 9);

  checkCosCyles(schedule);
}

}  // namespace thirdai::bolt::tests