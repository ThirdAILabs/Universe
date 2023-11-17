#include "Date.h"
#include <cereal/archives/binary.hpp>
#include <cereal/types/base_class.hpp>
#include <archive/src/Archive.h>
#include <archive/src/Map.h>
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

uint32_t dayOfWeek(const TimeObject& time) {
  return (time.secondsSinceEpoch() / TimeObject::SECONDS_IN_DAY) % 7;
}

uint32_t weekOfMonth(const TimeObject& time) {
  return time.dayOfMonthZeroIndexed() / 7;
}

uint32_t weekOfYear(const TimeObject& time) { return time.dayOfYear() / 7; }

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

#pragma omp parallel for default(none) \
    shared(dates, date_attributes, error) if (columns.numRows() > 1)
  for (size_t i = 0; i < dates->numRows(); i++) {
    try {
      TimeObject time(dates->value(i), _format);

      uint32_t day_of_week = dayOfWeek(time);

      uint32_t month = time.month();
      month += DAYS_IN_WEEK;

      uint32_t week_of_month = weekOfMonth(time);
      week_of_month += DAYS_IN_WEEK + MONTHS_IN_YEAR;

      uint32_t week_of_year = weekOfYear(time);
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

void Date::buildExplanationMap(const ColumnMap& input, State& state,
                               ExplanationMap& explanation) const {
  (void)state;

  auto output = apply(input, state);

  const auto& time_str =
      input.getValueColumn<std::string>(_input_column_name)->value(0);
  TimeObject time(
      input.getValueColumn<std::string>(_input_column_name)->value(0), _format);

  auto date_attributes =
      output.getArrayColumn<uint32_t>(_output_column_name)->row(0);

  std::string origin = explanation.explain(_input_column_name, time_str);

  explanation.store(_output_column_name, date_attributes[0],
                    "day of the week = " + std::to_string(dayOfWeek(time)) +
                        " from " + origin);

  explanation.store(_output_column_name, date_attributes[1],
                    "month of the year = " + std::to_string(time.month()) +
                        " from " + origin);

  explanation.store(_output_column_name, date_attributes[2],
                    "week of the month = " + std::to_string(weekOfMonth(time)) +
                        " from " + origin);

  explanation.store(_output_column_name, date_attributes[3],
                    "week of the year = " + std::to_string(weekOfYear(time)) +
                        " from " + origin);
}

ar::ConstArchivePtr Date::toArchive() const {
  auto map = ar::Map::make();

  map->set("type", ar::str(type()));
  map->set("input_column", ar::str(_input_column_name));
  map->set("output_column", ar::str(_output_column_name));
  map->set("format", ar::str(_format));

  return map;
}

template void Date::serialize(cereal::BinaryInputArchive&);
template void Date::serialize(cereal::BinaryOutputArchive&);

template <class Archive>
void Date::serialize(Archive& archive) {
  archive(cereal::base_class<Transformation>(this), _input_column_name,
          _output_column_name, _format);
}

}  // namespace thirdai::data

CEREAL_REGISTER_TYPE(thirdai::data::Date)