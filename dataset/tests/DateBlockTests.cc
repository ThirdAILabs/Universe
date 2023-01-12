#include <bolt_vector/src/BoltVector.h>
#include <gtest/gtest.h>
#include <dataset/src/batch_processors/GenericBatchProcessor.h>
#include <dataset/src/blocks/Date.h>
#include <sstream>

namespace thirdai::dataset {

class DateBlockTests : public testing::Test {
 protected:
  static auto featurize(const std::vector<std::string>& input_rows) {
    GenericBatchProcessor processor(
        /* input_blocks = */ {std::make_shared<DateBlock>(/* col = */ 0)},
        /* label_blocks = */ {});
    return processor.createBatch(input_rows).at(0);
  }

  static std::optional<uint32_t> dayOfWeek(const BoltVector& vector) {
    for (size_t act_idx = 0; act_idx < vector.len; act_idx++) {
      auto active_neuron = vector.active_neurons[act_idx];
      if (active_neuron >= dayOfWeekOffset() &&
          active_neuron < dayOfWeekOffset() + dayOfWeekDim()) {
        return active_neuron - dayOfWeekOffset();
      }
    }
    return {};
  }

  static std::optional<uint32_t> monthOfYear(const BoltVector& vector) {
    for (size_t act_idx = 0; act_idx < vector.len; act_idx++) {
      auto active_neuron = vector.active_neurons[act_idx];
      if (active_neuron >= monthOfYearOffset() &&
          active_neuron < monthOfYearOffset() + monthOfYearDim()) {
        return active_neuron - monthOfYearOffset();
      }
    }
    return {};
  }

  static std::optional<uint32_t> weekOfMonth(const BoltVector& vector) {
    for (size_t act_idx = 0; act_idx < vector.len; act_idx++) {
      auto active_neuron = vector.active_neurons[act_idx];
      if (active_neuron >= weekOfMonthOffset() &&
          active_neuron < weekOfMonthOffset() + weekOfMonthDim()) {
        return active_neuron - weekOfMonthOffset();
      }
    }
    return {};
  }

  static std::optional<uint32_t> weekOfYear(const BoltVector& vector) {
    for (size_t act_idx = 0; act_idx < vector.len; act_idx++) {
      auto active_neuron = vector.active_neurons[act_idx];
      if (active_neuron >= weekOfYearOffset() &&
          active_neuron < weekOfYearOffset() + weekOfYearDim()) {
        return active_neuron - weekOfYearOffset();
      }
    }
    return {};
  }

  static constexpr uint32_t dayOfWeekDim() {
    return DateBlock::day_of_week_dim;
  }
  static constexpr uint32_t monthOfYearDim() {
    return DateBlock::month_of_year_dim;
  }
  static constexpr uint32_t weekOfMonthDim() {
    return DateBlock::week_of_month_dim;
  }
  static constexpr uint32_t weekOfYearDim() {
    return DateBlock::week_of_year_dim;
  }

  static constexpr uint32_t dayOfWeekOffset() { return 0; }
  static constexpr uint32_t monthOfYearOffset() {
    return dayOfWeekOffset() + dayOfWeekDim();
  }
  static constexpr uint32_t weekOfMonthOffset() {
    return monthOfYearOffset() + monthOfYearDim();
  }
  static constexpr uint32_t weekOfYearOffset() {
    return weekOfMonthOffset() + weekOfMonthDim();
  }
};

TEST_F(DateBlockTests, ThrowsErrorGivenBadlyFormattedDate) {
  std::vector<std::vector<std::string>> samples = {
      {"20000101"},
      {"2000%01%01"},
  };
  for (auto& sample : samples) {
    ASSERT_THROW(  // NOLINT since clang-tidy doesn't like ASSERT_THROW
        featurize(sample), std::invalid_argument);
  }
}

TEST_F(DateBlockTests, ValidValues) {
  std::vector<std::string> samples = {
      "1900-01-01", "2001-01-14", "2005-02-29", "2030-06-22", "2100-12-31",
  };

  auto batch = featurize(samples);
  for (const auto& vec : batch) {
    ASSERT_EQ(vec.len, 4);

    auto day_of_week = dayOfWeek(vec);
    ASSERT_TRUE(day_of_week.has_value());
    ASSERT_GE(*day_of_week, 0);
    ASSERT_LT(*day_of_week, dayOfWeekDim());

    auto month_of_year = monthOfYear(vec);
    ASSERT_TRUE(month_of_year.has_value());
    ASSERT_GE(*month_of_year, 0);
    ASSERT_LT(*month_of_year, monthOfYearDim());

    auto week_of_month = weekOfMonth(vec);
    ASSERT_TRUE(week_of_month.has_value());
    ASSERT_GE(*week_of_month, 0);
    ASSERT_LT(*week_of_month, weekOfMonthDim());

    auto week_of_year = weekOfYear(vec);
    ASSERT_TRUE(week_of_year.has_value());
    ASSERT_GE(*week_of_year, 0);
    ASSERT_LT(*week_of_year, weekOfYearDim());
  }
}

TEST_F(DateBlockTests, DayOfWeekBehavesAsExpected) {
  std::vector<bool> days_of_week_seen(dayOfWeekDim());
  std::vector<std::string> samples = {
      "2000-02-28", "2000-02-29", "2000-03-01", "2000-03-02",
      "2000-03-03", "2000-03-04", "2000-03-05",
  };
  auto batch = featurize(samples);

  for (const auto& vec : batch) {
    auto day_of_week = dayOfWeek(vec);
    ASSERT_TRUE(day_of_week.has_value());
    days_of_week_seen[*day_of_week] = true;
  }

  for (auto seen : days_of_week_seen) {
    ASSERT_TRUE(seen);
  }
}

TEST_F(DateBlockTests, MonthOfYearBehavesAsExpected) {
  std::vector<bool> months_of_year_seen(monthOfYearDim());
  std::vector<std::string> samples = {
      "2000-01-01", "2000-02-01", "2000-03-01", "2000-04-01",
      "2000-05-01", "2000-06-01", "2000-07-01", "2000-08-01",
      "2000-09-01", "2000-10-01", "2000-11-01", "2000-12-01",
  };
  auto batch = featurize(samples);

  for (const auto& vec : batch) {
    auto month_of_year = monthOfYear(vec);
    ASSERT_TRUE(month_of_year.has_value());
    months_of_year_seen[*month_of_year] = true;
  }

  for (auto seen : months_of_year_seen) {
    ASSERT_TRUE(seen);
  }
}

TEST_F(DateBlockTests, WeekOfMonthBehavesAsExpected) {
  std::vector<bool> weeks_of_month_seen(weekOfMonthDim());
  std::vector<std::string> samples = {
      "2000-01-01", "2000-01-09", "2000-01-19", "2000-01-22", "2000-01-29",
  };
  auto batch = featurize(samples);

  for (const auto& vec : batch) {
    auto week_of_month = weekOfMonth(vec);
    ASSERT_TRUE(week_of_month.has_value());
    weeks_of_month_seen[*week_of_month] = true;
  }

  for (auto seen : weeks_of_month_seen) {
    ASSERT_TRUE(seen);
  }
}

TEST_F(DateBlockTests, WeekOfYearBehavesAsExpected) {
  std::vector<bool> weeks_of_year_seen(weekOfMonthDim());

  std::vector<std::string> month_strs;
  for (uint32_t month = 1; month <= 12; month++) {
    std::stringstream month_ss;
    if (month < 10) {
      month_ss << "0";
    }
    month_ss << month;
    month_strs.push_back(month_ss.str());
  }

  std::vector<std::string> samples;
  for (uint32_t month = 1; month <= 9; month++) {
    samples.push_back("2000-" + month_strs[0] + "-01");
    samples.push_back("2000-" + month_strs[0] + "-09");
    samples.push_back("2000-" + month_strs[0] + "-19");
    samples.push_back("2000-" + month_strs[0] + "-22");
    samples.push_back("2000-" + month_strs[0] + "-29");
  }

  auto batch = featurize(samples);

  for (const auto& vec : batch) {
    auto week_of_year = weekOfYear(vec);
    ASSERT_TRUE(week_of_year.has_value());
    weeks_of_year_seen[*week_of_year] = true;
  }

  for (auto seen : weeks_of_year_seen) {
    ASSERT_TRUE(seen);
  }
}

TEST_F(DateBlockTests, ExplainMethodGivesCorrectDayName) {
  std::vector<std::string> samples = {
      "2000-02-28", "2000-02-29", "2000-03-01", "2000-03-02",
      "2000-03-03", "2000-03-04", "2000-03-05",
  };
  std::vector<std::string> days = {
      "Monday", "Tuesday",  "Wednesday", "Thursday",
      "Friday", "Saturday", "Sunday",
  };
  auto block = std::make_shared<DateBlock>(/* col = */ 0);
  auto batch = featurize(samples);

  uint32_t day_number = 0;

  for (const auto& vec : batch) {
    auto day_of_week = dayOfWeek(vec);
    ASSERT_TRUE(day_of_week.has_value());
    auto day = block->explainIndex(*day_of_week, {});
    ASSERT_TRUE(days[day_number++] == day.keyword);
  }
}

TEST_F(DateBlockTests, ExplainMethodGivesCorrectMonthName) {
  std::vector<std::string> samples = {
      "2000-01-17", "2000-02-28", "2000-03-29", "2000-04-01",
      "2000-05-02", "2000-06-03", "2000-07-04", "2000-08-05",
      "2000-09-19", "2000-10-24", "2000-11-27", "2000-12-14",
  };
  std::vector<std::string> months = {
      "January", "February", "March",     "April",   "May",      "June",
      "July",    "August",   "September", "October", "November", "December",
  };
  auto block = std::make_shared<DateBlock>(/* col = */ 0);
  auto batch = featurize(samples);

  uint32_t month_number = 0;

  for (const auto& vec : batch) {
    auto month_of_year = monthOfYear(vec);
    ASSERT_TRUE(month_of_year.has_value());
    auto month = block->explainIndex(*month_of_year + monthOfYearOffset(), {});
    ASSERT_TRUE(months[month_number++] == month.keyword);
  }
}

}  // namespace thirdai::dataset
