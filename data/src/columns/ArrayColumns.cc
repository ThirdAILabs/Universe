#include "ArrayColumns.h"
#include "ColumnUtils.h"
#include <data/src/columns/Column.h>
#include <algorithm>
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

template <typename T>
ColumnPtr ArrayColumn<T>::concat(ColumnPtr&& other) {
  if (_dimension != other->dimension()) {
    throw std::invalid_argument(
        "Can only concatenate columns with the same dimension.");
  }

  auto other_concrete = std::dynamic_pointer_cast<ArrayColumn<T>>(other);
  if (!other_concrete) {
    throw std::invalid_argument(
        "Can only concatenate value columns of the same type.");
  }

  auto new_data =
      concatVectors(std::move(_data), std::move(other_concrete->_data));

  auto new_column =
      ArrayColumnPtr<T>(new ArrayColumn<T>(std::move(new_data), _dimension));

  _dimension.reset();
  other_concrete->_dimension.reset();

  return new_column;
}

template <typename T>
std::pair<ColumnPtr, ColumnPtr> ArrayColumn<T>::split(size_t starting_offset) {
  auto [front, back] = splitVector(std::move(_data), starting_offset);

  auto front_col =
      ArrayColumnPtr<T>(new ArrayColumn<T>(std::move(front), _dimension));
  auto back_col =
      ArrayColumnPtr<T>(new ArrayColumn<T>(std::move(back), _dimension));

  _dimension.reset();

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
      new ArrayColumn<uint32_t>(std::move(data), ColumnDimension::sparse(dim)));
}

template <>
ArrayColumnPtr<float> ArrayColumn<float>::make(
<<<<<<< HEAD
    std::vector<std::vector<float>>&& data) {
  std::optional<ColumnDimension> dimension = std::nullopt;
  if (!data.empty()) {
    size_t dim = data.front().size();

    bool all_dims_match = std::all_of(
        data.begin(), data.end(),
        [dim](const std::vector<float>& row) { return row.size() == dim; });

    // For a dense column there can only be a dimension if all of the columns
    // have the same length.
    if (all_dims_match) {
      dimension = ColumnDimension::dense(dim);
    }
=======
    std::vector<std::vector<float>>&& data, std::optional<size_t> dim) {
  std::optional<ColumnDimension> dimension = std::nullopt;
  if (dim) {
    bool all_dims_match = std::all_of(
        data.begin(), data.end(),
        [dim](const std::vector<float>& row) { return row.size() == *dim; });

    if (!all_dims_match) {
      throw std::invalid_argument(
          "Not all rows in DecimalArray column match provided dimension.");
    }

    dimension = ColumnDimension::dense(*dim);
>>>>>>> be2a0b0c4b6c69d3931eee4322d3a202017ebc8c
  }

  return ArrayColumnPtr<float>(
      new ArrayColumn<float>(std::move(data), dimension));
}

}  // namespace thirdai::data