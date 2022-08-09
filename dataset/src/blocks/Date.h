#pragma once

#include "BlockInterface.h"
#include <dataset/src/utils/TimeUtils.h>

namespace thirdai::dataset {

class DateBlockTests;

class DateBlock : public Block {
  friend DateBlockTests;

 public:
  explicit DateBlock(uint32_t col) : _col(col) {}

  uint32_t featureDim() const final {
    return day_of_week_dim + month_of_year_dim + week_of_month_dim +
           week_of_year_dim;
  };

  bool isDense() const final { return false; };

  uint32_t expectedNumColumns() const final { return _col + 1; };

 protected:
  static constexpr uint32_t day_of_week_dim = 7;
  static constexpr uint32_t month_of_year_dim = 12;
  static constexpr uint32_t week_of_month_dim = 5;
  static constexpr uint32_t week_of_year_dim = 49;

  std::exception_ptr buildSegment(
      const std::vector<std::string_view>& input_row,
      SegmentedFeatureVector& vec) final {
    
    struct tm time;
    auto exception = TimeUtils::timeStringToTimeObject(input_row[_col], time);
    if (exception) {
      return exception;
    }

    uint32_t epoch_time = TimeUtils::timeToEpoch(time);
    uint32_t offset = 0;

    vec.addSparseFeatureToSegment(offset + dayOfWeek(epoch_time), 1.0);
    offset += day_of_week_dim;

    vec.addSparseFeatureToSegment(offset + time.tm_mon, 1.0);
    offset += month_of_year_dim;

    vec.addSparseFeatureToSegment(offset + weekOfMonth(time), 1.0);
    offset += week_of_month_dim;

    vec.addSparseFeatureToSegment(offset + weekOfYear(time), 1.0);

    return nullptr;
  }

 private:
  static constexpr uint32_t dayOfWeek(uint32_t epoch_time) {
    return (epoch_time / TimeUtils::SECONDS_IN_DAY) % 7;
  }
  
  static constexpr uint32_t weekOfMonth(struct tm& time) {
    return (time.tm_mday - 1) / 7; // tm_mday starts at 1.
  }

  static constexpr uint32_t weekOfYear(struct tm& time) {
    uint32_t weeks_to_current_month = time.tm_mon * 4;
    return weeks_to_current_month + weekOfMonth(time);
  }
  
  uint32_t _col;
};

}  // namespace thirdai::dataset