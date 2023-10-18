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
                                std::optional<size_t> dim = std::nullopt);

  size_t numRows() const final { return _data.size(); }

  std::optional<size_t> dim() const final { return _dim; }

  RowView<T> row(size_t i) const final {
    if (i >= _data.size()) {
      throw std::out_of_range("ArrayColumn::row");
    }
    return {_data[i].data(), _data[i].size()};
  }

  void setRow(size_t row, ColumnRow new_row) final {
    _data[row].clear();
    auto rowArray = new_row.getArray<T>();
    _data[row].insert(_data[row].begin(), rowArray.begin(), rowArray.end());
  }

  void shuffle(const std::vector<size_t>& permutation) final;

  ColumnPtr permute(const std::vector<size_t>& permutation) const final;

  ColumnPtr concat(ColumnPtr&& other) final;

  std::pair<ColumnPtr, ColumnPtr> split(size_t starting_offset) final;

  const auto& data() const { return _data; }

 private:
  ArrayColumn(std::vector<std::vector<T>>&& data, std::optional<size_t> dim)
      : _data(std::move(data)), _dim(dim) {}

  std::vector<std::vector<T>> _data;
  std::optional<size_t> _dim;
};

}  // namespace thirdai::data