#include <gtest/gtest.h>
#include <dataset/src/utils/CountHistory.h>
#include <dataset/src/utils/TimeUtils.h>
#include <sys/types.h>
#include <string>
#include <unordered_map>

namespace thirdai::dataset {

TEST(UserCountHistoryTest, SanityCheck) {

  UserCountHistory history(/* lookahead_periods= */ 1, /* lookback_periods= */ 5);

  std::string user = "user";
  int64_t query_timestamp = 5 * TimeObject::SECONDS_IN_DAY;
  
  for (auto count : history.getHistory(user, query_timestamp)) {
    ASSERT_EQ(count, 0.0);
  }

  for (int64_t day = 0; day < 5; day++) {
    int64_t timestamp = day * TimeObject::SECONDS_IN_DAY;
    history.index(user, timestamp, static_cast<float>(day));
  }

  auto last_5_days = history.getHistory(user, query_timestamp);
  for (uint32_t day = 0; day < 5; day++) {
    ASSERT_EQ(last_5_days[day], static_cast<float>(day));
  }
}

TEST(UserCountHistoryTest, DifferentUsersMapToDifferentCounts) {

  UserCountHistory history(/* lookahead_periods= */ 0, /* lookback_periods= */ 1);

  history.index("user_1", /* timestamp= */ 0, /* val= */ 1.0);
  history.index("user_2", /* timestamp= */ 0, /* val= */ 20.0);
  history.index("user_3", /* timestamp= */ 0, /* val= */ 36.0);

  ASSERT_EQ(history.getHistory("user_1", /* timestamp= */ 0)[0], 1.0);
  ASSERT_EQ(history.getHistory("user_2", /* timestamp= */ 0)[0], 20.0);
  ASSERT_EQ(history.getHistory("user_3", /* timestamp= */ 0)[0], 36.0);
}

TEST(UserCountHistoryTest, DifferentPeriodsMapToDifferentCounts) {

  UserCountHistory history(/* lookahead_periods= */ 0, /* lookback_periods= */ 1, /* period_seconds= */ 1);

  history.index("user_1", /* timestamp= */ 0, /* val= */ 1.0);
  history.index("user_1", /* timestamp= */ 1, /* val= */ 20.0);
  history.index("user_1", /* timestamp= */ 2, /* val= */ 36.0);

  ASSERT_EQ(history.getHistory("user_1", /* timestamp= */ 0)[0], 1.0);
  ASSERT_EQ(history.getHistory("user_1", /* timestamp= */ 1)[0], 20.0);
  ASSERT_EQ(history.getHistory("user_1", /* timestamp= */ 2)[0], 36.0);
}

static float computeCountHistoryErrorWithVariableSketchSize(uint32_t sketch_rows, uint32_t sketch_range) {
  UserCountHistory history(/* lookahead_periods= */ 0, /* lookback_periods= */ 1, /* period_seconds= */ 1, sketch_rows, sketch_range);
  for (uint32_t user_id = 0; user_id < 1000; user_id++) {
    for (int64_t timestamp = 0; timestamp < 1000; timestamp++) {
      history.index(std::to_string(user_id), timestamp, 1.0);
    }
  }

  float error = 0.0;

  for (uint32_t user_id = 0; user_id < 1000; user_id++) {
    for (int64_t timestamp = 0; timestamp < 1000; timestamp++) {
      auto recent_history = history.getHistory(std::to_string(user_id), timestamp);
      EXPECT_EQ(recent_history.size(), 1);
      EXPECT_GE(recent_history[0], 1.0);
      error += recent_history[0] - 1.0;
    }
  }
  return error;
}

TEST(UserCountHistoryTest, ErrorDecreasesWithMoreSketchRows) {
  float errorWithOneRow = computeCountHistoryErrorWithVariableSketchSize(/* sketch_rows= */ 1, /* sketch_range= */ 1 << 22);
  float errorWithFiveRows = computeCountHistoryErrorWithVariableSketchSize(/* sketch_rows= */ 5, /* sketch_range= */ 1 << 22);
  float errorWithTenRows = computeCountHistoryErrorWithVariableSketchSize(/* sketch_rows= */ 10, /* sketch_range= */ 1 << 22);

  ASSERT_GT(errorWithOneRow, errorWithFiveRows);
  ASSERT_GT(errorWithFiveRows, errorWithTenRows);
}

TEST(UserCountHistoryTest, ErrorDecreasesWithWiderSketchRange) {
  float errorWithNarrowRange = computeCountHistoryErrorWithVariableSketchSize(/* sketch_rows= */ 5, /* sketch_range= */ 1 << 16);
  float errorWithMediumRange = computeCountHistoryErrorWithVariableSketchSize(/* sketch_rows= */ 5, /* sketch_range= */ 1 << 19);
  float errorWithWideRange = computeCountHistoryErrorWithVariableSketchSize(/* sketch_rows= */ 5, /* sketch_range= */ 1 << 22);

  ASSERT_GT(errorWithNarrowRange, errorWithMediumRange);
  ASSERT_GT(errorWithMediumRange, errorWithWideRange);
}

static void indexMockSamples(UserCountHistory& history) {
  for (int64_t period = 1; period <= 5; period++) {
    int64_t timestamp = period * UserCountHistory::DEFAULT_PERIOD_SECONDS;
    history.index("user_1", timestamp, /* val= */ period);
  }
}

TEST(UserCountHistoryTest, CorrectlyHandlesLookaheads) {
  int64_t query_timestamp = static_cast<int64_t>(5) * UserCountHistory::DEFAULT_PERIOD_SECONDS;
  
  UserCountHistory history_lookahead_0(/* lookahead_periods= */ 0, /* lookback_periods= */ 1);
  indexMockSamples(history_lookahead_0);
  ASSERT_EQ(history_lookahead_0.getHistory("user_1", query_timestamp)[0], 5.0);
  
  UserCountHistory history_lookahead_1(/* lookahead_periods= */ 1, /* lookback_periods= */ 1);
  indexMockSamples(history_lookahead_1);
  ASSERT_EQ(history_lookahead_1.getHistory("user_1", query_timestamp)[0], 4.0);
  
  UserCountHistory history_lookahead_2(/* lookahead_periods= */ 2, /* lookback_periods= */ 1);
  indexMockSamples(history_lookahead_2);
  ASSERT_EQ(history_lookahead_2.getHistory("user_1", query_timestamp)[0], 3.0);
}

TEST(UserCountHistoryTest, CorrectlyHandlesLookbacks) {
  int64_t query_timestamp = static_cast<int64_t>(5) * UserCountHistory::DEFAULT_PERIOD_SECONDS;
  
  UserCountHistory history_lookback_1(/* lookahead_periods= */ 0, /* lookback_periods= */ 1);
  indexMockSamples(history_lookback_1);
  auto recent_history_lookback_1 = history_lookback_1.getHistory("user_1", query_timestamp);
  ASSERT_EQ(recent_history_lookback_1.size(), 1);
  ASSERT_EQ(recent_history_lookback_1[0], 5.0);
  
  UserCountHistory history_lookback_2(/* lookahead_periods= */ 0, /* lookback_periods= */ 2);
  indexMockSamples(history_lookback_2);
  auto recent_history_lookback_2 = history_lookback_2.getHistory("user_1", query_timestamp);
  ASSERT_EQ(recent_history_lookback_2.size(), 2);
  ASSERT_EQ(recent_history_lookback_2[0], 4.0);
  ASSERT_EQ(recent_history_lookback_2[1], 5.0);
  
  UserCountHistory history_lookback_3(/* lookahead_periods= */ 0, /* lookback_periods= */ 3);
  indexMockSamples(history_lookback_3);
  auto recent_history_lookback_3 = history_lookback_3.getHistory("user_1", query_timestamp);
  ASSERT_EQ(recent_history_lookback_3.size(), 3);
  ASSERT_EQ(recent_history_lookback_3[0], 3.0);
  ASSERT_EQ(recent_history_lookback_3[1], 4.0);
  ASSERT_EQ(recent_history_lookback_3[2], 5.0);
}

TEST(UserCountHistoryTest, CorrectlyHandlesDifferentPeriods) {
  int64_t query_timestamp = static_cast<int64_t>(5) * UserCountHistory::DEFAULT_PERIOD_SECONDS;
  
  uint32_t default_period = UserCountHistory::DEFAULT_PERIOD_SECONDS;
  UserCountHistory history_period_default(/* lookahead_periods= */ 0, /* lookback_periods= */ 1, /* period_seconds= */ default_period);
  indexMockSamples(history_period_default);
  ASSERT_EQ(history_period_default.getHistory("user_1", query_timestamp)[0], 5.0);
  
  
  UserCountHistory history_period_2x_default(/* lookahead_periods= */ 0, /* lookback_periods= */ 1, /* period_seconds= */ 2 * default_period);
  indexMockSamples(history_period_2x_default);
  ASSERT_EQ(history_period_2x_default.getHistory("user_1", query_timestamp)[0], 9.0);

  UserCountHistory history_period_3x_default(/* lookahead_periods= */ 0, /* lookback_periods= */ 1, /* period_seconds= */ 3 * default_period);
  indexMockSamples(history_period_3x_default);
  ASSERT_EQ(history_period_3x_default.getHistory("user_1", query_timestamp)[0], 12.0);
}

TEST(UserCountHistoryTest, CorrectlyRemovesOutdatedCounts) {
  UserCountHistory history(/* lookahead_periods= */ 5, /* lookback_periods= */ 5);
  int64_t query_timestamp = static_cast<int64_t>(5) * UserCountHistory::DEFAULT_PERIOD_SECONDS;

  history.removeOutdatedCounts(/* timestamp= */ 0);
  history.index("user", /* timestamp= */ 0, 1.0);
  ASSERT_EQ(history.getHistory("user", query_timestamp)[4], 1.0);

  history.removeOutdatedCounts(/* timestamp= */ static_cast<int64_t>(10) * UserCountHistory::DEFAULT_PERIOD_SECONDS);
  ASSERT_EQ(history.getHistory("user", query_timestamp)[4], 1.0);
  
  history.removeOutdatedCounts(/* timestamp= */ static_cast<int64_t>(11) * UserCountHistory::DEFAULT_PERIOD_SECONDS);
  history.removeOutdatedCounts(/* timestamp= */ static_cast<int64_t>(22) * UserCountHistory::DEFAULT_PERIOD_SECONDS);
  ASSERT_EQ(history.getHistory("user", query_timestamp)[4], 0.0);
}

} // namespace thirdai::dataset

/**
 * 
 * Removing outdated sketch works check
 * 
 */



