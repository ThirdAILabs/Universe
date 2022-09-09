#pragma once

#include "BlockInterface.h"
#include <dataset/src/utils/TimeUtils.h>
#include <exception>
#include <stdexcept>

namespace thirdai::dataset {

class DateBlockTests;

/**
 * Parses a date string in "YYYY-MM-DD" format
 * and encodes in a vector the day of the week,
 * the month of the year, the week of the month,
 * and the week of the year, each one-hot
 * encoded in a separate segment of the vector.
 *
 * TODO(Geordie): try out other features
 * such as day of month, day of year, year, etc.
 */
class DateBlock : public Block {
  friend DateBlockTests;

 public:
  explicit DateBlock(uint32_t col) : _col(col) {}

  uint32_t featureDim() const final {
    return day_of_week_dim + month_of_year_dim + week_of_month_dim +
           week_of_year_dim;
  };

  bool isDense() const final { return false; };

  uint32_t expectedNumColumns() const final {
    // _col is the column index, so we expect at least
    // _col + 1 columns in each input row.
    return _col + 1;
  };

  std::pair<std::string, std::string> explainIndex(
      uint32_t index,
      std::optional<std::unordered_map<uint32_t, std::string>> num_to_name)
      const final {
    std::string response;
    if (index > 77) {
      throw std::invalid_argument(
          "index should not increase more than in date block.");
    }
    if (index < 7) {
      response = "day_of_week";
    } else if (7 <= index && index < 19) {
      response = "month_of_year";
    } else if (19 <= index && index < 24) {
      response = "week_of_month";
    } else {
      response = "week_of_year";
    }
    return std::make_pair(num_to_name->at(_col), response);
  }

 protected:
  static constexpr uint32_t day_of_week_dim = 7;
  static constexpr uint32_t month_of_year_dim = 12;
  static constexpr uint32_t week_of_month_dim = 5;
  static constexpr uint32_t week_of_year_dim = 53;

  std::exception_ptr buildSegment(
      const std::vector<std::string_view>& input_row,
      SegmentedFeatureVector& vec, bool store_map) final {
    (void)store_map;
    TimeObject time;

    try {
      time = TimeObject(input_row[_col]);
    } catch (const std::invalid_argument& e) {
      return std::make_exception_ptr(e);
    }

    uint32_t epoch_time = time.secondsSinceEpoch();
    uint32_t offset = 0;

    uint32_t day_of_week = (epoch_time / TimeObject::SECONDS_IN_DAY) % 7;
    vec.addSparseFeatureToSegment(offset + day_of_week, 1.0);
    offset += day_of_week_dim;

    vec.addSparseFeatureToSegment(offset + time.month(), 1.0);
    offset += month_of_year_dim;

    uint32_t week_of_month = time.dayOfMonthZeroIndexed() / 7;
    vec.addSparseFeatureToSegment(offset + week_of_month, 1.0);
    offset += week_of_month_dim;

    uint32_t week_of_year = time.dayOfYear() / 7;
    vec.addSparseFeatureToSegment(offset + week_of_year, 1.0);

    return nullptr;
  }

 private:
  uint32_t _col;
};

}  // namespace thirdai::dataset