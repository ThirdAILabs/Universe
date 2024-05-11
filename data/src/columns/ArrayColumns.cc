#include "ArrayColumns.h"
#include "ColumnUtils.h"
#include <data/src/columns/Column.h>
#include <algorithm>
#include <cstddef>
#include <memory>
#include <optional>
#include <stdexcept>

namespace thirdai::data {

template <typename T>
void ArrayColumn<T>::shuffle(const std::vector<size_t>& permutation) {
  _data = shuffleVector(std::move(_data), permutation);
}

template class ArrayColumn<uint32_t>;
template class ArrayColumn<float>;
template class ArrayColumn<std::string>;

template <typename T>
ColumnPtr ArrayColumn<T>::concat(ColumnPtr&& other) {
  if (dim() != other->dim()) {
    throw std::invalid_argument(
        "Can only concatenate columns with the same dimension.");
  }

  auto other_array_col = ArrayColumnBase<T>::cast(other);
  if (!other_array_col) {
    throw std::invalid_argument(
        "Can only concatenate array columns of the same type.");
  }

  // If the other column is an ArrayColumn, we can make use of the common
  // storage type to avoid copies.
  if (auto other_concrete = std::dynamic_pointer_cast<ArrayColumn<T>>(other)) {
    auto new_data =
        concatVectors(std::move(_data), std::move(other_concrete->_data));

    auto dim = _dim;
    _dim.reset();
    other_concrete->_dim.reset();

    return ArrayColumnPtr<T>(new ArrayColumn<T>(std::move(new_data), dim));
  }

  // If the other column only inherits from ArrayColumnBase, we need to copy
  // it's data.
  std::vector<std::vector<T>> new_data = std::move(_data);
  new_data.reserve(numRows() + other_array_col->numRows());

  for (size_t i = 0; i < other->numRows(); i++) {
    auto row = other_array_col->row(i);
    new_data.emplace_back(row.begin(), row.end());
  }

  auto dim = _dim;
  _dim.reset();

  return ArrayColumnPtr<T>(new ArrayColumn<T>(std::move(new_data), dim));
}

template <typename T>
std::pair<ColumnPtr, ColumnPtr> ArrayColumn<T>::split(size_t starting_offset) {
  auto [front, back] = splitVector(std::move(_data), starting_offset);

  auto front_col =
      ArrayColumnPtr<T>(new ArrayColumn<T>(std::move(front), dim()));
  auto back_col = ArrayColumnPtr<T>(new ArrayColumn<T>(std::move(back), dim()));

  _dim.reset();

  return {front_col, back_col};
}

template <>
ArrayColumnPtr<uint32_t> ArrayColumn<uint32_t>::make(
    std::vector<std::vector<uint32_t>>&& data, std::optional<size_t> dim) {
  if (dim) {
    for (const auto& row : data) {
      for (uint32_t index : row) {
        if (index >= *dim) {
          throw std::invalid_argument("Invalid index " + std::to_string(index) +
                                      " for TokenArrayColumn with dimension " +
                                      std::to_string(*dim) + ".");
        }
      }
    }
  }

  return ArrayColumnPtr<uint32_t>(
      new ArrayColumn<uint32_t>(std::move(data), dim));
}

template <>
ArrayColumnPtr<float> ArrayColumn<float>::make(
    std::vector<std::vector<float>>&& data, std::optional<size_t> dim) {
  if (dim) {
    bool all_dims_match = std::all_of(
        data.begin(), data.end(),
        [dim](const std::vector<float>& row) { return row.size() == *dim; });

    if (!all_dims_match) {
      throw std::invalid_argument(
          "Not all rows in DecimalArray column match provided dimension.");
    }
  }

  return ArrayColumnPtr<float>(new ArrayColumn<float>(std::move(data), dim));
}

template <>
ArrayColumnPtr<std::string> ArrayColumn<std::string>::make(
    std::vector<std::vector<std::string>>&& data, std::optional<size_t> dim) {
  if (dim) {
    bool all_dims_match = std::all_of(
        data.begin(), data.end(), [dim](const std::vector<std::string>& row) {
          return row.size() == *dim;
        });

    if (!all_dims_match) {
      throw std::invalid_argument(
          "Not all rows in StringArray column match provided dimension.");
    }
  }
  return ArrayColumnPtr<std::string>(
      new ArrayColumn<std::string>(std::move(data), dim));
}

template <typename T>
ColumnPtr ArrayColumn<T>::permute(
    const std::vector<size_t>& permutation) const {
  auto new_data = permuteVector(_data, permutation);
  return ArrayColumnPtr<T>(new ArrayColumn<T>(std::move(new_data), _dim));
}

}  // namespace thirdai::data