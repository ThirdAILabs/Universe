#include <gtest/gtest.h>
#include <dataset/src/utils/QuantityHistoryTracker.h>
#include <dataset/src/utils/TimeUtils.h>
#include <sys/types.h>
#include <optional>
#include <string>
#include <unordered_map>

namespace thirdai::dataset {

TEST(QuantityHistoryTrackerTest, SanityCheck) {
  QuantityHistoryTracker history(/* history_lag= */ 1,
                                 /* history_length= */ 5);

  ASSERT_EQ(history.historyLength(), 5);

  std::string key = "key";
  int64_t query_timestamp = 5 * TimeObject::SECONDS_IN_DAY;

  for (auto count : history.getHistory(key, query_timestamp)) {
    ASSERT_EQ(count, 0.0);
  }

  for (int64_t day = 0; day < 5; day++) {
    int64_t timestamp = day * TimeObject::SECONDS_IN_DAY;
    history.index(key, timestamp, static_cast<float>(day));
  }

  auto last_5_days = history.getHistory(key, query_timestamp);
  for (uint32_t day = 0; day < 5; day++) {
    ASSERT_EQ(last_5_days[day], static_cast<float>(day));
  }
}

TEST(QuantityHistoryTrackerTest, DifferentKeysMapToDifferentCounts) {
  QuantityHistoryTracker history(/* history_lag= */ 0,
                                 /* history_length= */ 1);

  history.index("key_1", /* timestamp= */ 0, /* val= */ 1.0);
  history.index("key_2", /* timestamp= */ 0, /* val= */ 20.0);
  history.index("key_3", /* timestamp= */ 0, /* val= */ 36.0);

  ASSERT_EQ(history.getHistory("key_1", /* current_timestamp= */ 0)[0], 1.0);
  ASSERT_EQ(history.getHistory("key_2", /* current_timestamp= */ 0)[0], 20.0);
  ASSERT_EQ(history.getHistory("key_3", /* current_timestamp= */ 0)[0], 36.0);
}

TEST(QuantityHistoryTrackerTest, DifferentPeriodsMapToDifferentCounts) {
  QuantityHistoryTracker history(
      /* history_lag= */ 0, /* history_length= */ 1,
      /* tracking_granularity= */ QuantityTrackingGranularity::Daily);

  history.index("key_1", /* timestamp= */ 0, /* val= */ 1.0);
  history.index("key_1", /* timestamp= */ 1 * TimeObject::SECONDS_IN_DAY,
                /* val= */ 20.0);
  history.index("key_1", /* timestamp= */ 2 * TimeObject::SECONDS_IN_DAY,
                /* val= */ 36.0);

  ASSERT_EQ(history.getHistory("key_1", /* current_timestamp= */ 0)[0], 1.0);
  ASSERT_EQ(history.getHistory("key_1", /* current_timestamp= */ 1 *
                                            TimeObject::SECONDS_IN_DAY)[0],
            20.0);
  ASSERT_EQ(history.getHistory("key_1", /* current_timestamp= */ 2 *
                                            TimeObject::SECONDS_IN_DAY)[0],
            36.0);
}

static std::pair<float, uint32_t>
computeCountHistoryErrorWithVariableSketchSize(
    uint32_t sketch_rows, uint32_t sketch_range,
    std::optional<float> expected_error = std::nullopt) {
  QuantityHistoryTracker history(
      /* history_lag= */ 0, /* history_length= */ 1,
      /* tracking_granularity= */ QuantityTrackingGranularity::Daily,
      sketch_rows, sketch_range);
  for (uint32_t key_id = 0; key_id < 500; key_id++) {
    for (int64_t day = 0; day < 1000; day++) {
      history.index(std::to_string(key_id), day * TimeObject::SECONDS_IN_DAY,
                    1.0);
    }
  }

  float error = 0.0;
  uint32_t times_under_expected_error = 0;

  for (uint32_t key_id = 0; key_id < 500; key_id++) {
    for (int64_t day = 0; day < 1000; day++) {
      auto recent_history = history.getHistory(
          std::to_string(key_id), day * TimeObject::SECONDS_IN_DAY);
      EXPECT_EQ(recent_history.size(), 1);
      EXPECT_GE(recent_history[0], 1.0);
      error += recent_history[0] - 1.0;
      if (expected_error) {
        EXPECT_LT(recent_history[0] - 1.0, *expected_error * 2);
        if (recent_history[0] - 1.0 <= *expected_error) {
          times_under_expected_error++;
        }
      }
    }
  }
  return {error, times_under_expected_error};
}

TEST(QuantityHistoryTrackerTest, ErrorDecreasesWithMoreSketchRows) {
  /*
    Theoretical error rate of count min sketch:
    For each count,
    with probability 1 - 1/(e^rows),
    error <= e / range * (sum of counts) = e / (2^16) * 500,000 = 21.0
  */
  auto [error_with_one_row, times_under_expected_error_one_row] =
      computeCountHistoryErrorWithVariableSketchSize(
          /* sketch_rows= */ 1, /* sketch_range= */ 1 << 16,
          /* expected_error= */ 21.0);
  auto [error_with_five_rows, times_under_expected_error_five_rows] =
      computeCountHistoryErrorWithVariableSketchSize(
          /* sketch_rows= */ 5, /* sketch_range= */ 1 << 16,
          /* expected_error= */ 21.0);

  ASSERT_GT(error_with_one_row, error_with_five_rows);

  // Probability of error <= 36 = 1 - 1/e = 63%. Make this lenient -> 50%.
  ASSERT_GT(times_under_expected_error_one_row,
            0.5 * 500000);  // 500,000 samples.

  // Probability of error <= 36 = 1 - 1/e^5 = 99%. Make this lenient -> 80%.
  ASSERT_GT(times_under_expected_error_five_rows,
            0.8 * 500000);  // 500,000 samples.
}

TEST(QuantityHistoryTrackerTest, ErrorDecreasesWithWiderSketchRange) {
  auto [error_with_narrow_range, _0] =
      computeCountHistoryErrorWithVariableSketchSize(
          /* sketch_rows= */ 5, /* sketch_range= */ 1 << 16);
  auto [error_with_medium_range, _1] =
      computeCountHistoryErrorWithVariableSketchSize(
          /* sketch_rows= */ 5, /* sketch_range= */ 1 << 19);
  auto [error_with_wide_range, _2] =
      computeCountHistoryErrorWithVariableSketchSize(
          /* sketch_rows= */ 5, /* sketch_range= */ 1 << 22);

  ASSERT_GT(error_with_narrow_range, error_with_medium_range);
  ASSERT_GT(error_with_medium_range, error_with_wide_range);
}

static void indexMockSamples(QuantityHistoryTracker& history) {
  for (int64_t period = 1; period <= 5; period++) {
    int64_t timestamp = period * TimeObject::SECONDS_IN_DAY;
    history.index("key_1", timestamp, /* val= */ period);
  }
}

TEST(QuantityHistoryTrackerTest, CorrectlyHandlesHistoryLags) {
  int64_t query_timestamp =
      static_cast<int64_t>(5) * TimeObject::SECONDS_IN_DAY;

  QuantityHistoryTracker history_lag_0(/* history_lag= */ 0,
                                       /* history_length= */ 1);
  indexMockSamples(history_lag_0);
  ASSERT_EQ(history_lag_0.getHistory("key_1", query_timestamp)[0], 5.0);

  QuantityHistoryTracker history_lag_1(/* history_lag= */ 1,
                                       /* history_length= */ 1);
  indexMockSamples(history_lag_1);
  ASSERT_EQ(history_lag_1.getHistory("key_1", query_timestamp)[0], 4.0);

  QuantityHistoryTracker history_lag_2(/* history_lag= */ 2,
                                       /* history_length= */ 1);
  indexMockSamples(history_lag_2);
  ASSERT_EQ(history_lag_2.getHistory("key_1", query_timestamp)[0], 3.0);
}

TEST(QuantityHistoryTrackerTest, CorrectlyHandlesHistoryLengths) {
  int64_t query_timestamp =
      static_cast<int64_t>(5) * TimeObject::SECONDS_IN_DAY;

  QuantityHistoryTracker history_length_1(/* history_lag= */ 0,
                                          /* history_length= */ 1);
  indexMockSamples(history_length_1);
  auto recent_history_length_1 =
      history_length_1.getHistory("key_1", query_timestamp);
  ASSERT_EQ(recent_history_length_1.size(), 1);
  ASSERT_EQ(recent_history_length_1[0], 5.0);

  QuantityHistoryTracker history_length_2(/* history_lag= */ 0,
                                          /* history_length= */ 2);
  indexMockSamples(history_length_2);
  auto recent_history_length_2 =
      history_length_2.getHistory("key_1", query_timestamp);
  ASSERT_EQ(recent_history_length_2.size(), 2);
  ASSERT_EQ(recent_history_length_2[0], 4.0);
  ASSERT_EQ(recent_history_length_2[1], 5.0);

  QuantityHistoryTracker history_length_3(/* history_lag= */ 0,
                                          /* history_length= */ 3);
  indexMockSamples(history_length_3);
  auto recent_history_length_3 =
      history_length_3.getHistory("key_1", query_timestamp);
  ASSERT_EQ(recent_history_length_3.size(), 3);
  ASSERT_EQ(recent_history_length_3[0], 3.0);
  ASSERT_EQ(recent_history_length_3[1], 4.0);
  ASSERT_EQ(recent_history_length_3[2], 5.0);
}

TEST(QuantityHistoryTrackerTest, CorrectlyHandlesDifferentPeriods) {
  int64_t query_timestamp =
      static_cast<int64_t>(5) * TimeObject::SECONDS_IN_DAY;

  QuantityHistoryTracker history_period_daily(
      /* history_lag= */ 0,
      /* history_length= */ 1,
      /* tracking_granularity= */ QuantityTrackingGranularity::Daily);
  indexMockSamples(history_period_daily);
  ASSERT_EQ(history_period_daily.getHistory("key_1", query_timestamp)[0], 5.0);

  QuantityHistoryTracker history_period_weekly(
      /* history_lag= */ 0, /* history_length= */ 1,
      /* tracking_granularity= */ QuantityTrackingGranularity::Weekly);
  indexMockSamples(history_period_weekly);
  ASSERT_EQ(history_period_weekly.getHistory("key_1", query_timestamp)[0],
            15.0);
}

TEST(QuantityHistoryTrackerTest, CorrectlyRemovesOutdatedCounts) {
  QuantityHistoryTracker history(/* history_lag= */ 5,
                                 /* history_length= */ 5);
  int64_t query_timestamp =
      static_cast<int64_t>(5) * TimeObject::SECONDS_IN_DAY;

  history.checkpointCurrentTimestamp(/* timestamp= */ 0);
  history.index("key", /* timestamp= */ 0, 1.0);
  ASSERT_EQ(history.getHistory("key", query_timestamp)[4], 1.0);

  history.checkpointCurrentTimestamp(/* timestamp= */ static_cast<int64_t>(10) *
                                     TimeObject::SECONDS_IN_DAY);
  ASSERT_EQ(history.getHistory("key", query_timestamp)[4], 1.0);

  history.checkpointCurrentTimestamp(/* timestamp= */ static_cast<int64_t>(11) *
                                     TimeObject::SECONDS_IN_DAY);
  history.checkpointCurrentTimestamp(/* timestamp= */ static_cast<int64_t>(22) *
                                     TimeObject::SECONDS_IN_DAY);
  ASSERT_EQ(history.getHistory("key", query_timestamp)[4], 0.0);
}

}  // namespace thirdai::dataset
