#include "ValueColumns.h"
#include "ColumnUtils.h"
#include <stdexcept>

namespace thirdai::data {

template <typename T>
void ValueColumn<T>::shuffle(const std::vector<size_t>& permutation) {
  _data = shuffleVector(std::move(_data), permutation);
}

template <typename T>
ColumnPtr ValueColumn<T>::concat(ColumnPtr&& other) {
  if (dim() != other->dim()) {
    throw std::invalid_argument(
        "Can only concatenate columns with the same dimension.");
  }

  auto other_concrete = std::dynamic_pointer_cast<ValueColumn<T>>(other);
  if (!other_concrete) {
    throw std::invalid_argument(
        "Can only concatenate value columns of the same type.");
  }

  auto new_data =
      concatVectors(std::move(_data), std::move(other_concrete->_data));

  auto new_column = std::shared_ptr<ValueColumn>(
      new ValueColumn<T>(std::move(new_data), _dim));

  _dim.reset();
  other_concrete->_dim.reset();

  return new_column;
}

template <typename T>
std::pair<ColumnPtr, ColumnPtr> ValueColumn<T>::split(size_t starting_offset) {
  auto [front, back] = splitVector(std::move(_data), starting_offset);

  auto front_col =
      ValueColumnPtr<T>(new ValueColumn<T>(std::move(front), _dim));
  auto back_col = ValueColumnPtr<T>(new ValueColumn<T>(std::move(back), _dim));

  _dim.reset();

  return {front_col, back_col};
}

template <>
ValueColumnPtr<uint32_t> ValueColumn<uint32_t>::make(
    std::vector<uint32_t>&& data, std::optional<size_t> dim) {
  if (dim) {
    for (uint32_t index : data) {
      if (index >= *dim) {
        throw std::invalid_argument("Invalid index " + std::to_string(index) +
                                    " for TokenColumn with dimension " +
                                    std::to_string(*dim) + ".");
      }
    }
  }

  return ValueColumnPtr<uint32_t>(
      new ValueColumn<uint32_t>(std::move(data), dim));
}

template <>
ValueColumnPtr<float> ValueColumn<float>::make(std::vector<float>&& data) {
  return ValueColumnPtr<float>(
      new ValueColumn<float>(std::move(data), /* dim= */ 1));
}

template <typename T>
ValueColumnPtr<T> ValueColumn<T>::make(std::vector<T>&& data) {
  return ValueColumnPtr<T>(new ValueColumn<T>(std::move(data), std::nullopt));
}

template ValueColumnPtr<std::string> ValueColumn<std::string>::make(
    std::vector<std::string>&&);

template ValueColumnPtr<int64_t> ValueColumn<int64_t>::make(
    std::vector<int64_t>&&);

template <typename T>
ColumnPtr ValueColumn<T>::permute(
    const std::vector<size_t>& permutation) const {
  auto new_data = permuteVector(_data, permutation);
  return ValueColumnPtr<T>(new ValueColumn<T>(std::move(new_data), _dim));
}

}  // namespace thirdai::data