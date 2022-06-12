#pragma once

#include "BlockInterface.h"
#include <dataset/src/utils/TimeUtils.h>

namespace thirdai::dataset {

class DateBlock : public Block {
 public:
  explicit DateBlock(uint32_t col) : _col(col) {}

  uint32_t featureDim() const final { 
    return day_of_week_dim + month_of_year_dim + week_of_month_dim + week_of_year_dim; 
  };

  bool isDense() const final { return false; };

  uint32_t expectedNumColumns() const final { return _col + 1; };

 protected:

  static constexpr uint32_t day_of_week_dim = 7;
  static constexpr uint32_t month_of_year_dim = 12;
  static constexpr uint32_t week_of_month_dim = 5;
  static constexpr uint32_t week_of_year_dim = 53;

  void buildSegment(const std::vector<std::string_view>& input_row,
                    SegmentedFeatureVector& vec) final {
    
    auto time = TimeUtils::timeStringToTimeObject(input_row[_col]);
    uint32_t offset = 0;

    // Day of week
    vec.addSparseFeatureToSegment(offset + time.tm_wday, 1.0);
    offset += day_of_week_dim;

    // Month of year
    vec.addSparseFeatureToSegment(offset + time.tm_mon, 1.0);
    offset += month_of_year_dim;

    // Week of month
    vec.addSparseFeatureToSegment(offset + (time.tm_mday - 1) / 7, 1.0);
    offset += week_of_month_dim;

    // Week of year
    vec.addSparseFeatureToSegment(offset + time.tm_yday / 7, 1.0);
  }

 private:
  uint32_t _col;
};

} // namespace thirdai::dataset