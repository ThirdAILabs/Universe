#pragma once

#include <data/src/columns/Column.h>
#include <optional>
#include <stdexcept>

namespace thirdai::data {

template <typename T>
class ArrayColumn;

template <typename T>
using ArrayColumnPtr = std::shared_ptr<ArrayColumn<T>>;

template <typename T>
class ArrayColumn : public ArrayColumnBase<T> {
 public:
  static ArrayColumnPtr<T> make(std::vector<std::vector<T>>&& data,
                                std::optional<size_t> dim);

  static ArrayColumnPtr<T> make(std::vector<std::vector<T>>&& data,
                                std::optional<ColumnDimension> dim) {
    if (!dim) {
      return make(std::move(data), std::nullopt);
    }
    return make(std::move(data), dim->dim);
  }

  size_t numRows() const final { return _data.size(); }

  std::optional<ColumnDimension> dimension() const final { return _dimension; }

  RowView<T> row(size_t i) const final {
    if (i >= _data.size()) {
      throw std::out_of_range("ArrayColumn::row");
    }
    return {_data[i].data(), _data[i].size()};
  }

  void shuffle(const std::vector<size_t>& permutation) final;

  ColumnPtr permute(const std::vector<size_t>& permutation) const final;

  ColumnPtr concat(ColumnPtr&& other) final;

  std::pair<ColumnPtr, ColumnPtr> split(size_t starting_offset) final;

  const auto& data() const { return _data; }

 private:
  ArrayColumn(std::vector<std::vector<T>>&& data,
              std::optional<ColumnDimension> dimension)
      : _data(std::move(data)), _dimension(dimension) {}

  std::vector<std::vector<T>> _data;
  std::optional<ColumnDimension> _dimension;
};

}  // namespace thirdai::data