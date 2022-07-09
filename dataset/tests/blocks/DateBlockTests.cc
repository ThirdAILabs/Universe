#include <gtest/gtest.h>
#include <dataset/src/blocks/BlockInterface.h>
#include <dataset/src/blocks/Date.h>
#include <dataset/src/utils/SegmentedFeatureVector.h>
#include <dataset/src/utils/TimeUtils.h>
#include "BlockTest.h"

namespace thirdai::dataset {

class DateBlockTests : public BlockTest {
 public:
  static StringMatrix mockDataset(uint32_t n_days) {
    StringMatrix matrix(n_days);
    TimestampGenerator time_gen("2020-01-01");
    for (uint32_t i = 0; i < n_days; i++) {
      matrix[i].push_back(time_gen.currentTimeString());
      time_gen.addDays(1);
    }
    return matrix;
  }

  static uint32_t dayOfWeekOffset() { return 0; }
  static uint32_t dayOfWeekDim() { return DateBlock::day_of_week_dim; }
  static uint32_t monthOfYearOffset() { return dayOfWeekOffset() + dayOfWeekDim(); }
  static uint32_t monthOfYearDim() { return DateBlock::month_of_year_dim; }
  static uint32_t weekOfMonthOffset() { return monthOfYearOffset() + monthOfYearDim(); }
  static uint32_t weekOfMonthDim() { return DateBlock::week_of_month_dim; }
  static uint32_t weekOfYearOffset() { return weekOfMonthOffset() + weekOfMonthDim(); }

};

TEST_F(DateBlockTests, CorrectOutput) {
  uint32_t n_days = 100;
  auto mock_data = mockDataset(n_days);
  DateBlock block(/* col = */ 0);
  std::vector<SegmentedSparseFeatureVector> vecs;
  for (auto& row : mock_data) {
    SegmentedSparseFeatureVector vec;
    addVectorSegmentWithBlock(block, row, vec);
    vecs.push_back(vec);
  }

  int32_t last_day_of_week = -1;
  int32_t last_week_of_month = -1;
  int32_t last_month_of_year = -1;
  int32_t last_week_of_year = -1;

  uint32_t day_of_week_changes = 0;
  uint32_t week_of_month_changes = 0;
  uint32_t month_of_year_changes = 0;
  uint32_t week_of_year_changes = 0;

  for (auto& vec : vecs) {
    uint32_t days_of_week_found = 0;
    uint32_t weeks_of_month_found = 0;
    uint32_t months_of_year_found = 0;
    uint32_t weeks_of_year_found = 0;
    auto entries = vectorEntries(vec);
    for (auto [k, _] : entries) {
      ASSERT_LT(k, block.featureDim());
      if (k < dayOfWeekDim()) {
        int32_t cur_day_of_week = k;
        if (cur_day_of_week != last_day_of_week) {
          day_of_week_changes++;
          last_day_of_week = cur_day_of_week;
        }
        days_of_week_found++;

      } else if (k < monthOfYearOffset() + monthOfYearDim()) {
        int32_t cur_month_of_year = k - monthOfYearOffset();
        if (cur_month_of_year != last_month_of_year) {
          month_of_year_changes++;
          last_month_of_year = cur_month_of_year;
        }
        months_of_year_found++;

      } else if (k < weekOfMonthOffset() + weekOfMonthDim()) {
        int32_t cur_week_of_month = k - weekOfMonthOffset();
        std::cout << cur_week_of_month << std::endl;
        if (cur_week_of_month != last_week_of_month) {
          week_of_month_changes++;
          last_week_of_month = cur_week_of_month;
        }
        weeks_of_month_found++;

      } else {
        int32_t cur_week_of_year = k - weekOfYearOffset();
        if (cur_week_of_year != last_week_of_year) {
          week_of_year_changes++;
          last_week_of_year = cur_week_of_year;
        }
        weeks_of_year_found++;
      }
    }

    ASSERT_EQ(days_of_week_found, 1);
    ASSERT_EQ(weeks_of_month_found, 1);
    ASSERT_EQ(months_of_year_found, 1);
    ASSERT_EQ(weeks_of_year_found, 1);
  }

  ASSERT_EQ(day_of_week_changes, n_days);
  ASSERT_GE(week_of_month_changes, n_days / 7);
  ASSERT_GE(month_of_year_changes, n_days / 30);
  ASSERT_GE(week_of_year_changes, n_days / 7);
}

} // namespace thirdai::dataset