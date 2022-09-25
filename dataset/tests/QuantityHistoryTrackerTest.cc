#include <gtest/gtest.h>
#include <dataset/src/utils/QuantityHistoryTracker.h>
#include <dataset/src/utils/TimeUtils.h>
#include <sys/types.h>
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

static float computeCountHistoryErrorWithVariableSketchSize(
    uint32_t sketch_rows, uint32_t sketch_range) {
  QuantityHistoryTracker history(
      /* history_lag= */ 0, /* history_length= */ 1,
      /* tracking_granularity= */ QuantityTrackingGranularity::Daily,
      sketch_rows, sketch_range);
  for (uint32_t key_id = 0; key_id < 1000; key_id++) {
    for (int64_t day = 0; day < 1000; day++) {
      history.index(std::to_string(key_id), day * TimeObject::SECONDS_IN_DAY,
                    1.0);
    }
  }

  float error = 0.0;

  for (uint32_t key_id = 0; key_id < 1000; key_id++) {
    for (int64_t day = 0; day < 1000; day++) {
      auto recent_history = history.getHistory(
          std::to_string(key_id), day * TimeObject::SECONDS_IN_DAY);
      EXPECT_EQ(recent_history.size(), 1);
      EXPECT_GE(recent_history[0], 1.0);
      error += recent_history[0] - 1.0;
    }
  }
  return error;
}

TEST(QuantityHistoryTrackerTest, ErrorDecreasesWithMoreSketchRows) {
  float errorWithOneRow = computeCountHistoryErrorWithVariableSketchSize(
      /* sketch_rows= */ 1, /* sketch_range= */ 1 << 22);
  float errorWithFiveRows = computeCountHistoryErrorWithVariableSketchSize(
      /* sketch_rows= */ 5, /* sketch_range= */ 1 << 22);
  float errorWithTenRows = computeCountHistoryErrorWithVariableSketchSize(
      /* sketch_rows= */ 10, /* sketch_range= */ 1 << 22);

  ASSERT_GT(errorWithOneRow, errorWithFiveRows);
  ASSERT_GT(errorWithFiveRows, errorWithTenRows);
}

TEST(QuantityHistoryTrackerTest, ErrorDecreasesWithWiderSketchRange) {
  float errorWithNarrowRange = computeCountHistoryErrorWithVariableSketchSize(
      /* sketch_rows= */ 5, /* sketch_range= */ 1 << 16);
  float errorWithMediumRange = computeCountHistoryErrorWithVariableSketchSize(
      /* sketch_rows= */ 5, /* sketch_range= */ 1 << 19);
  float errorWithWideRange = computeCountHistoryErrorWithVariableSketchSize(
      /* sketch_rows= */ 5, /* sketch_range= */ 1 << 22);

  ASSERT_GT(errorWithNarrowRange, errorWithMediumRange);
  ASSERT_GT(errorWithMediumRange, errorWithWideRange);
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
