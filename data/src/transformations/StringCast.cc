#include "StringCast.h"
#include <data/src/ColumnMap.h>
#include <data/src/columns/ArrayColumns.h>
#include <data/src/columns/ValueColumns.h>
#include <data/src/transformations/Transformation.h>
#include <dataset/src/utils/TimeUtils.h>
#include <utils/StringManipulation.h>
#include <exception>
#include <stdexcept>
#include <string>

namespace thirdai::data {

std::exception_ptr formatParseError(const std::string& row,
                                    const std::string& column) {
  return std::make_exception_ptr(std::invalid_argument(
      "Invalid row '" + row + "' in column '" + column + "'."));
}

template <>
CastToValue<uint32_t>::CastToValue(std::string input_column_name,
                                   std::string output_column_name,
                                   std::optional<size_t> dim)
    : _input_column_name(std::move(input_column_name)),
      _output_column_name(std::move(output_column_name)),
      _dim(dim) {}

template <>
CastToValue<float>::CastToValue(std::string input_column_name,
                                std::string output_column_name)
    : _input_column_name(std::move(input_column_name)),
      _output_column_name(std::move(output_column_name)) {}

template <>
CastToValue<int64_t>::CastToValue(std::string input_column_name,
                                  std::string output_column_name,
                                  std::string format)
    : _input_column_name(std::move(input_column_name)),
      _output_column_name(std::move(output_column_name)),
      _format(std::move(format)) {}

template <typename T>
ColumnMap CastToValue<T>::apply(ColumnMap columns, State& state) const {
  (void)state;

  auto str_column = columns.getValueColumn<std::string>(_input_column_name);

  std::vector<T> rows(str_column->numRows());

  std::exception_ptr error;

#pragma omp parallel for default(none) shared(str_column, rows, error)
  for (size_t i = 0; i < str_column->numRows(); i++) {
    try {
      rows[i] = parse(str_column->value(i));
    } catch (...) {
#pragma omp critical
      error = formatParseError(str_column->value(i), _input_column_name);
    }
  }

  if (error) {
    std::rethrow_exception(error);
  }

  auto output_column = makeColumn(std::move(rows));
  columns.setColumn(_output_column_name, std::move(output_column));

  return columns;
}

template <>
uint32_t CastToValue<uint32_t>::parse(const std::string& row) const {
  return std::stoul(row);
}

template <>
float CastToValue<float>::parse(const std::string& row) const {
  return std::stof(row);
}

template <>
int64_t CastToValue<int64_t>::parse(const std::string& row) const {
  return dataset::TimeObject(row, _format).secondsSinceEpoch();
}

template <>
ColumnPtr CastToValue<uint32_t>::makeColumn(
    std::vector<uint32_t>&& rows) const {
  return ValueColumn<uint32_t>::make(std::move(rows), _dim);
}

template <typename T>
ColumnPtr CastToValue<T>::makeColumn(std::vector<T>&& rows) const {
  return ValueColumn<T>::make(std::move(rows));
}

template class CastToValue<uint32_t>;
template class CastToValue<float>;
template class CastToValue<int64_t>;

template <typename T>
CastToArray<T>::CastToArray(std::string input_column_name,
                            std::string output_column_name, char delimiter,
                            std::optional<size_t> dim)
    : _input_column_name(std::move(input_column_name)),
      _output_column_name(std::move(output_column_name)),
      _delimiter(delimiter),
      _dim(dim) {}

template <typename T>
ColumnMap CastToArray<T>::apply(ColumnMap columns, State& state) const {
  (void)state;

  auto str_column = columns.getValueColumn<std::string>(_input_column_name);

  std::vector<std::vector<T>> rows(str_column->numRows());

  std::exception_ptr error;

#pragma omp parallel for default(none) shared(str_column, rows, error)
  for (size_t i = 0; i < str_column->numRows(); i++) {
    try {
      for (const auto& item : text::split(str_column->value(i), _delimiter)) {
        rows[i].push_back(parse(item));
      }
    } catch (...) {
#pragma omp critical
      error = formatParseError(str_column->value(i), _input_column_name);
    }
  }

  if (error) {
    std::rethrow_exception(error);
  }

  auto output_column = makeColumn(std::move(rows));
  columns.setColumn(_output_column_name, std::move(output_column));

  return columns;
}

template <>
uint32_t CastToArray<uint32_t>::parse(const std::string& row) const {
  return std::stoul(row);
}

template <>
float CastToArray<float>::parse(const std::string& row) const {
  return std::stof(row);
}

template <typename T>
ColumnPtr CastToArray<T>::makeColumn(std::vector<std::vector<T>>&& rows) const {
  return ArrayColumn<T>::make(std::move(rows), _dim);
}

template class CastToArray<uint32_t>;
template class CastToArray<float>;

}  // namespace thirdai::data
