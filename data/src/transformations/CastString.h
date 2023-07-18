#pragma once

#include <data/src/ColumnMap.h>
#include <data/src/columns/ArrayColumns.h>
#include <data/src/columns/Column.h>
#include <data/src/columns/Types.h>
#include <data/src/columns/ValueColumns.h>
#include <data/src/transformations/Transformation.h>
#include <dataset/src/utils/TimeUtils.h>
#include <utils/StringManipulation.h>
#include <algorithm>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>
namespace thirdai::data {

template <typename ColumnT, typename T>
class CastString : public Transformation {
 public:
  CastString(const std::vector<std::string>& input_columns,
             const std::vector<std::string>& output_columns)
      : _input_column_names(input_columns),
        _output_column_names(output_columns) {
    if (input_columns.size() != output_columns.size()) {
      throw std::invalid_argument(
          "Got different numbers of input and output columns. (" +
          std::to_string(input_columns.size()) + " vs " +
          std::to_string(output_columns.size()) + ")");
    }
  }

  ColumnMap apply(ColumnMap columns) const final {
    std::vector<std::vector<T>> data_for_outputs(
        _output_column_names.size(), std::vector<T>(columns.numRows()));

    std::vector<Ptr<StringColumn>> input_columns(_input_column_names.size());
    std::transform(_input_column_names.begin(), _input_column_names.end(),
                   input_columns.begin(), [&columns](const std::string& name) {
                     return columns.getValueColumn<std::string>(name);
                   });

#pragma omp parallel for default(none) \
    shared(columns, input_columns, data_for_outputs)
    for (uint32_t row = 0; row < columns.numRows(); ++row) {
      for (uint32_t col_id = 0; col_id < input_columns.size(); ++col_id) {
        data_for_outputs[col_id][row] =
            convert(input_columns[col_id]->value(row));
      }
    }

    for (uint32_t col_id = 0; col_id < data_for_outputs.size(); ++col_id) {
      columns.setColumn(
          /* name= */ _output_column_names[col_id],
          /* column= */ makeColumn(std::move(data_for_outputs[col_id])));
    }

    return columns;
  }

 private:
  virtual T convert(const std::string& original) const = 0;
  virtual Ptr<ColumnT> makeColumn(std::vector<T>&& data) const = 0;

  std::vector<std::string> _input_column_names;
  std::vector<std::string> _output_column_names;
};

template <typename ArrayColumnT, typename T>
class CastStringToArray : public CastString<ArrayColumnT, std::vector<T>> {
 public:
  CastStringToArray(const std::vector<std::string>& input_columns,
                    const std::vector<std::string>& output_columns,
                    char delimiter)
      : CastString<ArrayColumnT, std::vector<T>>(input_columns, output_columns),
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

class CastStringToToken final : public CastString<TokenColumn, uint32_t> {
 public:
  CastStringToToken(const std::vector<std::string>& input_columns,
                    const std::vector<std::string>& output_columns)
      : CastString<TokenColumn, uint32_t>(input_columns, output_columns) {}

 private:
  uint32_t convert(const std::string& original) const final {
    return text::toInteger(original.data());
  }

  Ptr<TokenColumn> makeColumn(std::vector<uint32_t>&& data) const final {
    return TokenColumn::make(std::move(data), std::nullopt);
  }
};

class CastStringToTokenArray final
    : public CastStringToArray<TokenArrayColumn, uint32_t> {
 public:
  CastStringToTokenArray(const std::vector<std::string>& input_columns,
                         const std::vector<std::string>& output_columns,
                         char delimiter)
      : CastStringToArray<TokenArrayColumn, uint32_t>(
            input_columns, output_columns, delimiter) {}

 private:
  uint32_t convertSingle(const std::string& single) const final {
    return text::toInteger(single.data());
  }

  Ptr<TokenArrayColumn> makeColumn(
      std::vector<std::vector<uint32_t>>&& data) const final {
    return TokenArrayColumn::make(std::move(data), std::nullopt);
  }
};

class CastStringToDecimal final : public CastString<DecimalColumn, float> {
 public:
  CastStringToDecimal(const std::vector<std::string>& input_columns,
                      const std::vector<std::string>& output_columns)
      : CastString<DecimalColumn, float>(input_columns, output_columns) {}

 private:
  float convert(const std::string& original) const final {
    return text::toInteger(original.data());
  }

  Ptr<DecimalColumn> makeColumn(std::vector<float>&& data) const final {
    return DecimalColumn::make(std::move(data));
  }
};

class CastStringToDecimalArray final
    : public CastStringToArray<DecimalArrayColumn, float> {
 public:
  CastStringToDecimalArray(const std::vector<std::string>& input_columns,
                           const std::vector<std::string>& output_columns,
                           char delimiter)
      : CastStringToArray<DecimalArrayColumn, float>(
            input_columns, output_columns, delimiter) {}

 private:
  float convertSingle(const std::string& single) const final {
    return std::stof(single);
  }

  Ptr<DecimalArrayColumn> makeColumn(
      std::vector<std::vector<float>>&& data) const final {
    return DecimalArrayColumn::make(std::move(data));
  }
};

class CastStringToTimestamp final
    : public CastString<TimestampColumn, int64_t> {
 public:
  CastStringToTimestamp(const std::vector<std::string>& input_columns,
                        const std::vector<std::string>& output_columns,
                        std::string format = "%Y-%m-%d")
      : CastString<TimestampColumn, int64_t>(input_columns, output_columns),
        _format(std::move(format)) {}

 private:
  int64_t convert(const std::string& single) const final {
    return dataset::TimeObject(single, _format).secondsSinceEpoch();
  }

  Ptr<TimestampColumn> makeColumn(std::vector<int64_t>&& data) const final {
    return TimestampColumn::make(std::move(data));
  }

  std::string _format;
};

}  // namespace thirdai::data