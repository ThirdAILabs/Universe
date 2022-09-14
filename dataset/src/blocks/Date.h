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

  ResponsibleColumnAndInputKey explainFeature(
      uint32_t index_within_block,
      std::optional<std::unordered_map<uint32_t, std::string>> num_to_name)
      const final {
    if (num_to_name == std::nullopt) {
      throw std::invalid_argument(
          "map of col num to col name is missing in date block.");
    }
    std::string response;
    if (index_within_block >= featureDim()) {
      throw std::invalid_argument("index is out of bounds for date block.");
    }
    if (index_within_block < day_of_week_dim) {
      response = "day_of_week";
    } else if (index_within_block < (day_of_week_dim + month_of_year_dim)) {
      response = "month_of_year";
    } else if (index_within_block <
               (day_of_week_dim + month_of_year_dim + week_of_month_dim)) {
      response = "week_of_month";
    } else {
      response = "week_of_year";
    }
    return {num_to_name->at(_col), response};
  }

 protected:
  static constexpr uint32_t day_of_week_dim = 7;
  static constexpr uint32_t month_of_year_dim = 12;
  static constexpr uint32_t week_of_month_dim = 5;
  static constexpr uint32_t week_of_year_dim = 53;

  std::exception_ptr buildSegment(
      const std::vector<std::string_view>& input_row,
      SegmentedFeatureVector& vec, bool remember_raw_features) final {
    (void)remember_raw_features;
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