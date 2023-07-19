#pragma once

#include <data/src/columns/Column.h>
#include <dataset/src/featurizers/ProcessorUtils.h>
#include <dataset/src/utils/CsvParser.h>
#include <cctype>
#include <cstdlib>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string_view>

namespace thirdai::data {

template <typename T>
class ValueColumn;

template <typename T>
using ValueColumnPtr = std::shared_ptr<ValueColumn<T>>;

template <typename T>
class ValueColumn : public ValueColumnBase<T> {
  friend ValueColumnPtr<uint32_t> makeTokenColumn(std::vector<uint32_t>&&,
                                                  std::optional<size_t>);

  friend ValueColumnPtr<float> makeDecimalColumn(std::vector<float>&&);

  friend ValueColumnPtr<std::string> makeStringColumn(
      std::vector<std::string>&&);

  friend ValueColumnPtr<int64_t> makeTimestampColumn(std::vector<int64_t>&&);

 public:
  size_t numRows() const final { return _data.size(); }

  std::optional<ColumnDimension> dimension() const final { return _dimension; }

  const T& value(size_t i) const final {
    if (i >= _data.size()) {
      throw std::out_of_range("CppValueColumn::value");
    }
    return _data[i];
  }

  RowView<T> row(size_t i) const final {
    if (i >= _data.size()) {
      throw std::out_of_range("CppValueColumn::row");
    }
    return {_data.data() + i, 1};
  }

  void shuffle(const std::vector<size_t>& permutation) final;

  std::shared_ptr<Column> concat(std::shared_ptr<Column>&& other) final;

  const auto& data() const { return _data; }

 private:
  ValueColumn(std::vector<T>&& data, std::optional<ColumnDimension> dimension)
      : _data(std::move(data)), _dimension(dimension) {}

  std::vector<T> _data;
  std::optional<ColumnDimension> _dimension;
};

ValueColumnPtr<uint32_t> makeTokenColumn(std::vector<uint32_t>&& data,
                                         std::optional<size_t> dim);

ValueColumnPtr<float> makeDecimalColumn(std::vector<float>&& data);

ValueColumnPtr<std::string> makeStringColumn(std::vector<std::string>&& data);

ValueColumnPtr<int64_t> makeTimestampColumn(std::vector<int64_t>&& data);

}  // namespace thirdai::data