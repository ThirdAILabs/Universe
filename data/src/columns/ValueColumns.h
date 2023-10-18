#pragma once

#include <data/src/columns/Column.h>
#include <cstddef>
#include <stdexcept>

namespace thirdai::data {

template <typename T>
class ValueColumn;

template <typename T>
using ValueColumnPtr = std::shared_ptr<ValueColumn<T>>;

template <typename T>
class ValueColumn : public ValueColumnBase<T> {
 public:
  static ValueColumnPtr<T> make(std::vector<T>&& data,
                                std::optional<size_t> dim);

  static ValueColumnPtr<T> make(std::vector<T>&& data);

  size_t numRows() const final { return _data.size(); }

  std::optional<size_t> dim() const final { return _dim; }

  const T& value(size_t i) const final {
    if (i >= _data.size()) {
      throw std::out_of_range("ValueColumn::value");
    }
    return _data[i];
  }

  RowView<T> row(size_t i) const final {
    if (i >= _data.size()) {
      throw std::out_of_range("ValueColumn::row");
    }
    return {_data.data() + i, 1};
  }

  void setRow(size_t row, ColumnRow new_row) final {
    _data[row] = new_row.getValue<T>();
  }

  void shuffle(const std::vector<size_t>& permutation) final;

  ColumnPtr permute(const std::vector<size_t>& permutation) const final;

  ColumnPtr concat(ColumnPtr&& other) final;

  std::pair<ColumnPtr, ColumnPtr> split(size_t starting_offset) final;

  const auto& data() const { return _data; }

 private:
  ValueColumn(std::vector<T>&& data, std::optional<size_t> dim)
      : _data(std::move(data)), _dim(dim) {}

  std::vector<T> _data;
  std::optional<size_t> _dim;
};

}  // namespace thirdai::data