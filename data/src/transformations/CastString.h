#pragma once

#include <data/src/ColumnMap.h>
#include <data/src/columns/ArrayColumns.h>
#include <data/src/columns/Column.h>
#include <data/src/columns/ValueColumns.h>
#include <data/src/transformations/Transformation.h>
#include <dataset/src/utils/TimeUtils.h>
#include <utils/StringManipulation.h>
#include <algorithm>
#include <exception>
#include <optional>
#include <string>
#include <vector>

namespace thirdai::data {

template <typename T>
class CastString : public Transformation {
 public:
  CastString(std::string input_column, std::string output_column)
      : _input_column_name(std::move(input_column)),
        _output_column_name(std::move(output_column)) {}

  ColumnMap apply(ColumnMap columns, State& state) const final {
    (void)state;

    std::vector<T> output_data(columns.numRows());
    auto input_column = columns.getValueColumn<std::string>(_input_column_name);

#pragma omp parallel for default(none) \
    shared(columns, input_column, output_data)
    for (uint32_t row = 0; row < columns.numRows(); ++row) {
      output_data[row] = convert(input_column->value(row));
    }

    columns.setColumn(
        /* name= */ _output_column_name,
        /* column= */ makeColumn(std::move(output_data)));

    return columns;
  }

 private:
  virtual T convert(const std::string& original) const = 0;
  virtual ColumnPtr makeColumn(std::vector<T>&& data) const = 0;

  std::string _input_column_name;
  std::string _output_column_name;
};

template <typename T>
class CastStringToArray : public CastString<std::vector<T>> {
 public:
  CastStringToArray(std::string input_column, std::string output_column,
                    char delimiter)
      : CastString<std::vector<T>>(std::move(input_column),
                                   std::move(output_column)),
        _delimiter(delimiter) {}

 private:
  std::vector<T> convert(const std::string& original) const final {
    auto split = text::split(original, _delimiter);
    std::vector<T> converted(split.size());
    std::transform(
        split.begin(), split.end(), converted.begin(),
        [&](const std::string& single) { return convertSingle(single); });
    return converted;
  }

  virtual T convertSingle(const std::string& single) const = 0;

  char _delimiter;
};

class CastStringToToken final : public CastString<uint32_t> {
 public:
  CastStringToToken(std::string input_column, std::string output_column,
                    std::optional<uint32_t> dim)
      : CastString<uint32_t>(std::move(input_column), std::move(output_column)),
        _dim(dim) {}

 private:
  uint32_t convert(const std::string& original) const final {
    return std::stoul(original);
  }

  ColumnPtr makeColumn(std::vector<uint32_t>&& data) const final {
    return ValueColumn<uint32_t>::make(std::move(data), _dim);
  }

  std::optional<uint32_t> _dim;
};

class CastStringToTokenArray final : public CastStringToArray<uint32_t> {
 public:
  CastStringToTokenArray(std::string input_column, std::string output_column,
                         char delimiter, std::optional<uint32_t> dim)
      : CastStringToArray<uint32_t>(std::move(input_column),
                                    std::move(output_column), delimiter),
        _dim(dim) {}

 private:
  uint32_t convertSingle(const std::string& single) const final {
    return std::stoul(single);
  }

  ColumnPtr makeColumn(std::vector<std::vector<uint32_t>>&& data) const final {
    return ArrayColumn<uint32_t>::make(std::move(data), _dim);
  }

  std::optional<uint32_t> _dim;
};

class CastStringToDecimal final : public CastString<float> {
 public:
  CastStringToDecimal(std::string input_column, std::string output_column)
      : CastString<float>(std::move(input_column), std::move(output_column)) {}

 private:
  float convert(const std::string& original) const final {
    return std::stof(original);
  }

  ColumnPtr makeColumn(std::vector<float>&& data) const final {
    return ValueColumn<float>::make(std::move(data));
  }
};

class CastStringToDecimalArray final : public CastStringToArray<float> {
 public:
  CastStringToDecimalArray(std::string input_column, std::string output_column,
                           char delimiter)
      : CastStringToArray<float>(std::move(input_column),
                                 std::move(output_column), delimiter) {}

 private:
  float convertSingle(const std::string& single) const final {
    return std::stof(single);
  }

  ColumnPtr makeColumn(std::vector<std::vector<float>>&& data) const final {
    return ArrayColumn<float>::make(std::move(data));
  }
};

class CastStringToTimestamp final : public CastString<int64_t> {
 public:
  CastStringToTimestamp(std::string input_column, std::string output_column,
                        std::string format)
      : CastString<int64_t>(std::move(input_column), std::move(output_column)),
        _format(std::move(format)) {}

 private:
  int64_t convert(const std::string& single) const final {
    return dataset::TimeObject(single, _format).secondsSinceEpoch();
  }

  ColumnPtr makeColumn(std::vector<int64_t>&& data) const final {
    return ValueColumn<int64_t>::make(std::move(data));
  }

  std::string _format;
};

}  // namespace thirdai::data