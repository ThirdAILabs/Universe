#include "StringCast.h"
#include <data/src/ColumnMap.h>
#include <data/src/columns/ArrayColumns.h>
#include <data/src/columns/Column.h>
#include <data/src/columns/ValueColumns.h>
#include <data/src/transformations/Transformation.h>
#include <dataset/src/utils/TimeUtils.h>
#include <proto/string_cast.pb.h>
#include <utils/StringManipulation.h>
#include <exception>
#include <stdexcept>
#include <string>
#include <type_traits>

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

#pragma omp parallel for default(none) \
    shared(str_column, rows, error) if (columns.numRows() > 1)
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
  if (row.empty()) {
    return 0.0;  // Handles missing values in tabular datasets.
  }
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

template <>
void CastToValue<uint32_t>::buildExplanationMap(
    const ColumnMap& input, State& state, ExplanationMap& explanations) const {
  (void)state;

  const std::string& value =
      input.getValueColumn<std::string>(_input_column_name)->value(0);

  std::string explanation = "token " + value + " from " +
                            explanations.explain(_input_column_name, value);

  explanations.store(_output_column_name, parse(value), explanation);
}

template <>
void CastToValue<float>::buildExplanationMap(
    const ColumnMap& input, State& state, ExplanationMap& explanations) const {
  (void)state;

  const std::string& value =
      input.getValueColumn<std::string>(_input_column_name)->value(0);

  std::string explanation = "decimal " + value + " from " +
                            explanations.explain(_input_column_name, value);

  explanations.store(_output_column_name,
                     /* feature_index = */ 0, explanation);
}

template <>
void CastToValue<int64_t>::buildExplanationMap(
    const ColumnMap& input, State& state, ExplanationMap& explanations) const {
  (void)state;

  const std::string& value =
      input.getValueColumn<std::string>(_input_column_name)->value(0);

  std::string explanation = "timestamp " + value + " from " +
                            explanations.explain(_input_column_name, value);

  explanations.store(_output_column_name,
                     /* feature_index = */ 0, explanation);
}

template <typename T>
proto::data::Transformation* CastToValue<T>::toProto() const {
  auto* transformation = new proto::data::Transformation();

  auto* cast = transformation->mutable_string_cast();

  cast->set_target(protoTargetType());

  cast->set_input_column(_input_column_name);
  cast->set_output_column(_output_column_name);

  if (_dim) {
    cast->set_dim(*_dim);
  }

  if constexpr (std::is_same_v<decltype(_format), std::string>) {
    cast->set_format(_format);
  }

  return transformation;
}

template <>
proto::data::StringCast::TargetType CastToValue<uint32_t>::protoTargetType()
    const {
  return proto::data::StringCast::TargetType::StringCast_TargetType_TOKEN;
}

template <>
proto::data::StringCast::TargetType CastToValue<float>::protoTargetType()
    const {
  return proto::data::StringCast::TargetType::StringCast_TargetType_DECIMAL;
}

template <>
proto::data::StringCast::TargetType CastToValue<int64_t>::protoTargetType()
    const {
  return proto::data::StringCast::TargetType::StringCast_TargetType_TIMESTAMP;
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

#pragma omp parallel for default(none) \
    shared(str_column, rows, error) if (columns.numRows() > 1)
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

template <>
void CastToArray<uint32_t>::buildExplanationMap(
    const ColumnMap& input, State& state, ExplanationMap& explanations) const {
  (void)state;

  std::string input_str =
      input.getValueColumn<std::string>(_input_column_name)->value(0);

  for (const auto& item : text::split(input_str, _delimiter)) {
    std::string explanation =
        "token " + item + " from " +
        explanations.explain(_input_column_name, input_str);

    explanations.store(_output_column_name, parse(item), explanation);
  }
}

template <>
void CastToArray<float>::buildExplanationMap(
    const ColumnMap& input, State& state, ExplanationMap& explanations) const {
  (void)state;

  std::string input_str =
      input.getValueColumn<std::string>(_input_column_name)->value(0);

  size_t index = 0;
  for (const auto& item : text::split(input_str, _delimiter)) {
    std::string explanation =
        "decimal " + item + " from " +
        explanations.explain(_input_column_name, input_str);

    explanations.store(_output_column_name, index++, explanation);
  }
}

template <typename T>
proto::data::Transformation* CastToArray<T>::toProto() const {
  auto* transformation = new proto::data::Transformation();

  auto* cast = transformation->mutable_string_cast();

  cast->set_target(protoTargetType());

  cast->set_input_column(_input_column_name);
  cast->set_output_column(_output_column_name);
  cast->set_delimiter(_delimiter);

  if (_dim) {
    cast->set_dim(*_dim);
  }

  return transformation;
}

template <>
proto::data::StringCast::TargetType CastToArray<uint32_t>::protoTargetType()
    const {
  return proto::data::StringCast::TargetType::StringCast_TargetType_TOKEN_ARRAY;
}

template <>
proto::data::StringCast::TargetType CastToArray<float>::protoTargetType()
    const {
  return proto::data::StringCast::TargetType::
      StringCast_TargetType_DECIMAL_ARRAY;
}

template class CastToArray<uint32_t>;
template class CastToArray<float>;

char getDelimiter(const proto::data::StringCast& cast) {
  if (!cast.has_delimiter()) {
    throw std::runtime_error("Expected delimiter for cast to array column.");
  }
  return cast.delimiter();
}

std::string getFormat(const proto::data::StringCast& cast) {
  if (!cast.has_format()) {
    throw std::runtime_error(
        "Expected time format for cast to timestamp column.");
  }
  return cast.format();
}

TransformationPtr stringCastFromProto(const proto::data::StringCast& cast) {
  std::optional<size_t> dim;
  if (cast.has_dim()) {
    dim = cast.dim();
  }

  switch (cast.target()) {
    case proto::data::StringCast::TOKEN:
      return std::make_shared<StringToToken>(cast.input_column(),
                                             cast.output_column(), dim);

    case proto::data::StringCast::TOKEN_ARRAY:
      return std::make_shared<StringToTokenArray>(
          cast.input_column(), cast.output_column(), getDelimiter(cast), dim);

    case proto::data::StringCast::DECIMAL:
      return std::make_shared<StringToDecimal>(cast.input_column(),
                                               cast.output_column());

    case proto::data::StringCast::DECIMAL_ARRAY:
      return std::make_shared<StringToDecimalArray>(
          cast.input_column(), cast.output_column(), getDelimiter(cast), dim);

    case proto::data::StringCast::TIMESTAMP:
      return std::make_shared<StringToTimestamp>(
          cast.input_column(), cast.output_column(), getFormat(cast));

    default:
      throw std::runtime_error("Invalid string cast target type in fromProto.");
  }
}

}  // namespace thirdai::data
