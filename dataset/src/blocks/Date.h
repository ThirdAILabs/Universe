#pragma once

#include <cereal/access.hpp>
#include <cereal/types/base_class.hpp>
#include <cereal/types/polymorphic.hpp>
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
class DateBlock final : public Block {
  friend DateBlockTests;

 public:
  explicit DateBlock(ColumnIdentifier col) : _col(std::move(col)) {}

  void updateColumnNumbers(const ColumnNumberMap& column_number_map) final {
    _col.updateColumnNumber(column_number_map);
  }

  bool hasColumnNames() const final { return _col.hasName(); }

  bool hasColumnNumbers() const final { return _col.hasNumber(); }

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

  Explanation explainIndex(uint32_t index_within_block,
                           const RowInput& input_row) final {
    return {_col.number(), getExplanationReason(index_within_block, input_row)};
  }

  Explanation explainIndex(uint32_t index_within_block,
                           const MapInput& input_map) final {
    return {_col.name(), getExplanationReason(index_within_block, input_map)};
  }

  template <typename ColumnarInputType>
  std::string getExplanationReason(uint32_t index_within_block,
                                   const ColumnarInputType& input) {
    (void)input;
    std::string reason;
    if (index_within_block >= featureDim()) {
      throw std::invalid_argument("index is out of bounds for date block.");
    }
    if (index_within_block < day_of_week_dim) {
      // The code gets the day number by dividing the unix timestamp by the
      // number of seconds in a day, then taking the modulo of that number
      // with 7. this unix timestamp starts from january 1, 1970 which is
      // thursday so, our day 0 is thursday and day 1 is friday etc..
      // so as we want getDayOfWeek function to be general that means we
      // normally assume sunday as 0 and so on.., so added 4 and took the
      // remainder with 7.
      reason = getDayOfWeek((index_within_block + 4) % 7);
    } else if (index_within_block < (day_of_week_dim + month_of_year_dim)) {
      reason = getMonthOfYear(index_within_block - day_of_week_dim);
    } else if (index_within_block <
               (day_of_week_dim + month_of_year_dim + week_of_month_dim)) {
      reason = "week_of_month";
    } else {
      reason = "week_of_year";
    }
    return reason;
  }

  static auto make(ColumnIdentifier col) {
    return std::make_shared<DateBlock>(std::move(col));
  }

 protected:
  static constexpr uint32_t day_of_week_dim = 7;
  static constexpr uint32_t month_of_year_dim = 12;
  static constexpr uint32_t week_of_month_dim = 5;
  static constexpr uint32_t week_of_year_dim = 53;

  std::exception_ptr buildSegment(const RowInput& input_row,
                                  SegmentedFeatureVector& vec) final {
    return buildSegmentImpl(input_row, vec);
  }

  std::exception_ptr buildSegment(const MapInput& input_map,
                                  SegmentedFeatureVector& vec) final {
    return buildSegmentImpl(input_map, vec);
  }

  template <typename ColumnarInputType>
  std::exception_ptr buildSegmentImpl(const ColumnarInputType& input,
                                      SegmentedFeatureVector& vec) {
    TimeObject time;

    try {
      time = TimeObject(getColumn(input, _col));
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
  ColumnIdentifier _col;

  // Constructor for Cereal
  DateBlock() {}

  friend class cereal::access;
  template <class Archive>
  void serialize(Archive& archive) {
    archive(cereal::base_class<Block>(this), _col);
  }
};

}  // namespace thirdai::dataset

CEREAL_REGISTER_TYPE(thirdai::dataset::DateBlock)