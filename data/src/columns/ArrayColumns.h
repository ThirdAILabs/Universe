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
class ArrayColumn;

template <typename T>
using ArrayColumnPtr = std::shared_ptr < ArrayColumn<T>>;

template <typename T>
class ArrayColumn : public ArrayColumnBase<T> {
  friend ArrayColumnPtr<uint32_t> makeTokenArrayColumn(
      std::vector<std::vector<uint32_t>>&&, std::optional<size_t>);

  friend ArrayColumnPtr<float> makeDecimalArrayColumn(
      std::vector<std::vector<float>>&&);

 public:
  size_t numRows() const final { return _data.size(); }

  std::optional<ColumnDimension> dimension() const final { return _dimension; }

  RowView<T> row(size_t i) const final {
    if (i >= _data.size()) {
      throw std::out_of_range("CppArrayColumn::row");
    }
    return {_data[i].data(), _data[i].size()};
  }

  void shuffle(const std::vector<size_t>& permutation) final;

  std::shared_ptr<Column> concat(std::shared_ptr<Column>&& other) final;

  const auto& data() const { return _data; }

 private:
  ArrayColumn(std::vector<std::vector<T>>&& data,
              std::optional<ColumnDimension> dimension)
      : _data(std::move(data)), _dimension(dimension) {}

  std::vector<std::vector<T>> _data;
  std::optional<ColumnDimension> _dimension;
};

ArrayColumnPtr<uint32_t> makeTokenArrayColumn(
    std::vector<std::vector<uint32_t>>&& data, std::optional<size_t> dim);

ArrayColumnPtr<float> makeDecimalArrayColumn(
    std::vector<std::vector<float>>&& data);

}  // namespace thirdai::data