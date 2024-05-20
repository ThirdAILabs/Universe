#include "StringCast.h"
#include <cereal/archives/binary.hpp>
#include <cereal/types/base_class.hpp>
#include <cereal/types/optional.hpp>
#include <cereal/types/polymorphic.hpp>
#include <archive/src/Archive.h>
#include <archive/src/Map.h>
#include <data/src/ColumnMap.h>
#include <data/src/columns/ArrayColumns.h>
#include <data/src/columns/ValueColumns.h>
#include <data/src/transformations/Transformation.h>
#include <dataset/src/utils/TimeUtils.h>
#include <utils/text/StringManipulation.h>
#include <exception>
#include <optional>
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
std::string typeName<uint32_t>() {
  return "u32";
}

template <>
std::string typeName<float>() {
  return "f32";
}

template <>
std::string typeName<int64_t>() {
  return "i64";
}

template <>
std::string typeName<std::string>() {
  return "str";
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

template <typename T>
ar::ConstArchivePtr CastToValue<T>::toArchive() const {
  auto map = ar::Map::make();

  map->set("type", ar::str(type()));
  map->set("input_column", ar::str(_input_column_name));
  map->set("output_column", ar::str(_output_column_name));
  if (_dim) {
    map->set("dim", ar::u64(*_dim));
  }
  if constexpr (std::is_same_v<T, int64_t>) {
    map->set("format", ar::str(_format));
  }

  return map;
}

template <typename T>
CastToValue<T>::CastToValue(const ar::Archive& archive)
    : _input_column_name(archive.str("input_column")),
      _output_column_name(archive.str("output_column")),
      _dim(archive.getOpt<ar::U64>("dim")) {
  if constexpr (std::is_same_v<T, int64_t>) {
    _format = archive.str("format");
  }
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
template <class Archive>
void CastToValue<T>::serialize(Archive& archive) {
  archive(cereal::base_class<Transformation>(this), _input_column_name,
          _output_column_name, _dim);
  if constexpr (std::is_same_v<T, int64_t>) {
    archive(_format);
  }
}

template void CastToValue<uint32_t>::serialize(cereal::BinaryInputArchive&);
template void CastToValue<uint32_t>::serialize(cereal::BinaryOutputArchive&);

template void CastToValue<float>::serialize(cereal::BinaryInputArchive&);
template void CastToValue<float>::serialize(cereal::BinaryOutputArchive&);

template void CastToValue<int64_t>::serialize(cereal::BinaryInputArchive&);
template void CastToValue<int64_t>::serialize(cereal::BinaryOutputArchive&);

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

template <typename T>
ar::ConstArchivePtr CastToArray<T>::toArchive() const {
  auto map = ar::Map::make();

  map->set("type", ar::str(type()));
  map->set("input_column", ar::str(_input_column_name));
  map->set("output_column", ar::str(_output_column_name));
  map->set("delimiter", ar::character(_delimiter));
  if (_dim) {
    map->set("dim", ar::u64(*_dim));
  }

  return map;
}

template <typename T>
CastToArray<T>::CastToArray(const ar::Archive& archive)
    : _input_column_name(archive.str("input_column")),
      _output_column_name(archive.str("output_column")),
      _delimiter(archive.getAs<ar::Char>("delimiter")),
      _dim(archive.getOpt<ar::U64>("dim")) {}

template <>
uint32_t CastToArray<uint32_t>::parse(const std::string& row) const {
  return std::stoul(row);
}

template <>
float CastToArray<float>::parse(const std::string& row) const {
  return std::stof(row);
}

template <>
std::string CastToArray<std::string>::parse(const std::string& row) const {
  return row;
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

template <>
void CastToArray<std::string>::buildExplanationMap(
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

template <typename T>
template <class Archive>
void CastToArray<T>::serialize(Archive& archive) {
  archive(cereal::base_class<Transformation>(this), _input_column_name,
          _output_column_name, _delimiter, _dim);
}

template void CastToArray<uint32_t>::serialize(cereal::BinaryInputArchive&);
template void CastToArray<uint32_t>::serialize(cereal::BinaryOutputArchive&);

template void CastToArray<float>::serialize(cereal::BinaryInputArchive&);
template void CastToArray<float>::serialize(cereal::BinaryOutputArchive&);

template void CastToArray<std::string>::serialize(cereal::BinaryInputArchive&);

template class CastToArray<uint32_t>;
template class CastToArray<float>;

}  // namespace thirdai::data

CEREAL_REGISTER_TYPE(thirdai::data::StringToToken)
CEREAL_REGISTER_TYPE(thirdai::data::StringToDecimal)
CEREAL_REGISTER_TYPE(thirdai::data::StringToDecimalArray)