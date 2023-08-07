#include "Date.h"
#include <data/src/columns/ArrayColumns.h>
#include <dataset/src/utils/TimeUtils.h>
#include <exception>
#include <string>

namespace thirdai::data {

static constexpr uint32_t DAYS_IN_WEEK = 7;
static constexpr uint32_t MONTHS_IN_YEAR = 12;
static constexpr uint32_t WEEKS_IN_MONTH = 5;
static constexpr uint32_t WEEKS_IN_YEAR = 53;

using dataset::TimeObject;

Date::Date(std::string input_column_name, std::string output_column_name,
           std::string format)
    : _input_column_name(std::move(input_column_name)),
      _output_column_name(std::move(output_column_name)),
      _format(std::move(format)) {}

ColumnMap Date::apply(ColumnMap columns, State& state) const {
  (void)state;

  auto dates = columns.getValueColumn<std::string>(_input_column_name);

  std::vector<std::vector<uint32_t>> date_attributes(columns.numRows());

  std::exception_ptr error;

#pragma omp parallel for default(none) shared(dates, date_attributes, error)
  for (size_t i = 0; i < dates->numRows(); i++) {
    try {
      TimeObject time(dates->value(i), _format);

      uint32_t day_of_week =
          (time.secondsSinceEpoch() / TimeObject::SECONDS_IN_DAY) % 7;

      uint32_t month = time.month();
      month += DAYS_IN_WEEK;

      uint32_t week_of_month = time.dayOfMonthZeroIndexed() / 7;
      week_of_month += DAYS_IN_WEEK + MONTHS_IN_YEAR;

      uint32_t week_of_year = time.dayOfYear() / 7;
      week_of_year += DAYS_IN_WEEK + MONTHS_IN_YEAR + WEEKS_IN_MONTH;

      date_attributes[i] = {day_of_week, month, week_of_month, week_of_year};
    } catch (...) {
#pragma omp critical
      error = std::current_exception();
    }
  }

  if (error) {
    std::rethrow_exception(error);
  }

  auto output = ArrayColumn<uint32_t>::make(
      std::move(date_attributes),
      DAYS_IN_WEEK + MONTHS_IN_YEAR + WEEKS_IN_MONTH + WEEKS_IN_YEAR);
  columns.setColumn(_output_column_name, output);

  return columns;
}

proto::data::Transformation* Date::toProto() const {
  auto* transformation = new proto::data::Transformation();
  auto* date = transformation->mutable_date();

  date->set_input_column(_input_column_name);
  date->set_output_column(_output_column_name);
  date->set_format(_format);

  return transformation;
}

}  // namespace thirdai::data